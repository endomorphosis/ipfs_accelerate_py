#!/usr/bin/env python3
"""ASEH-035 ContextPack reuse, omission, and net-savings benchmark.

Interface: ``AsehContextPackBenchmark@1``

Paired current-tree measures cover eligible reuse, before/after context
tokens, expansion precision/recall, critical omission detection, stale
rejection, build/retrieval cost, audit overhead, and net provider-cost
effect. Missing telemetry stays ``unavailable`` and is never numeric zero.
Hermetic results cannot grant live reuse or promotion. Seeded critical
omissions and stale packs are always rejected.
"""

from __future__ import annotations

import argparse
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
    PROGRAM_ID,
    SEALED_CURRENT_ARM_CONSTRAINTS,
    collect_unavailable_fields,
    content_identity,
    measured_quantity,
    unavailable,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
    EXACT_FRESHNESS_FIELDS,
    CurrentPackIdentity,
    StaleIdentityError,
    encode_context_pack_envelope,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack_selector import (
    select_current_minimal_pack,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.proof_context.context_pack import (
    CriticalOmissionError,
    build_minimal_semantic_pack,
)
from ipfs_datasets_py.proof_context.incremental_context import (
    expansion_precision_recall,
    expand_incremental_pack,
)
from ipfs_kit_py.proof_context.state_store import open_context_pack_store


PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
MANIFEST_PATH: Final[Path] = PACKAGE_DIR / "context_pack_manifest.json"

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
FIXTURE_TREE: Final[str] = "16ef68abe8a35a3033dfaf1ed4e8d6132600df8f"
STALE_TREE: Final[str] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TOOLCHAIN: Final[str] = "python3.12"
ENVIRONMENT: Final[str] = "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
SCHEMA_IDENTITY: Final[str] = "ipfs-datasets.proof-context.context-pack@0.1"
MILLIONTHS: Final[int] = 1_000_000
MAX_SERIALIZED_RECIPE_BYTES: Final[int] = 4_096
AUDIT_OVERHEAD_FLOOR: Final[int] = 1

REQUIRED_METRICS: Final[tuple[str, ...]] = (
    "eligible_reuse",
    "context_tokens_before",
    "context_tokens_after",
    "expansion_precision",
    "expansion_recall",
    "critical_omission_detection",
    "stale_pack_rejection",
    "build_cost",
    "retrieval_cost",
    "audit_overhead",
    "net_provider_cost_effect",
)

NAMED_MISSING_KINDS: Final[tuple[str, ...]] = (
    "cid",
    "symbol",
    "contract",
    "test",
    "counterexample",
    "obligation",
)

STALE_FIELDS: Final[tuple[str, ...]] = EXACT_FRESHNESS_FIELDS

ACCEPTANCE_TESTS: Final[dict[str, str]] = {
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


class ContextPackBenchmarkError(ValueError):
    """Closed ContextPack benchmark contract violation."""


class AggregateOnlyError(ContextPackBenchmarkError):
    """Campaign omitted per-fixture paired measures."""


class MissingAuditCostError(ContextPackBenchmarkError):
    """Audit overhead is missing, zero, or encoded as unavailable-as-zero."""


class AcceptedCriticalOmissionError(ContextPackBenchmarkError):
    """A seeded critical omission was admitted."""


class FixtureAsLiveError(ContextPackBenchmarkError):
    """A hermetic fixture was labeled live or granted live reuse."""


class StaleIdentityAdmissionError(ContextPackBenchmarkError):
    """A seeded stale pack was admitted."""


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


def _bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ContextPackBenchmarkError(f"{name} must be a boolean")
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


def cid_label(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def token_estimator() -> CalibratedTokenEstimator:
    return CalibratedTokenEstimator()


def estimate_tokens(payload: Mapping[str, Any] | bytes) -> int:
    estimator = token_estimator()
    if isinstance(payload, bytes):
        data = payload
    else:
        data = compact_json(payload).encode("utf-8")
    return _int(estimator.estimate(data), name="token_estimate", minimum=1)


def ratio_millionths(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return MILLIONTHS if numerator == 0 else 0
    return _int(
        (numerator * MILLIONTHS) // denominator,
        name="ratio_millionths",
        maximum=MILLIONTHS,
    )


def measured_count(value: int) -> dict[str, Any]:
    return measured_quantity(value, unit="count", sensor_id=SENSOR_ID)


def measured_tokens(value: int) -> dict[str, Any]:
    return measured_quantity(value, unit="tokens", sensor_id=SENSOR_ID)


def measured_compute(value: int) -> dict[str, Any]:
    return measured_quantity(value, unit="compute_units", sensor_id=SENSOR_ID)


def measured_ratio(value: int) -> dict[str, Any]:
    return measured_quantity(value, unit="ratio_millionths", sensor_id=SENSOR_ID)


def freshness_bindings(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "file_and_symbol_identities": [],
        "schema_identities": [SCHEMA_IDENTITY],
        "toolchain_identities": [TOOLCHAIN],
        "environment_requirements": [ENVIRONMENT],
        "reusable_until_conditions": ["tree-unchanged"],
    }
    payload.update(overrides)
    return payload


def helper_dependency(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "symbol": "helper",
        "cid": cid_label("helper"),
        "path": "helper.py",
        "meaning": "pure helper used by target",
    }
    item.update(overrides)
    return item


def unused_dependency(index: int = 0) -> dict[str, object]:
    suffix = "" if index == 0 else f"-{index:02d}"
    return {
        "symbol": f"unused-module{suffix}",
        "cid": cid_label(f"unused-module{suffix}"),
        "path": f"unused{suffix}.py",
        "meaning": "unreferenced whole-repository member",
    }


def named_catalog_record(kind: str, name: str) -> dict[str, object]:
    return {
        "kind": kind,
        "name": name,
        "cid": cid_label(f"{kind}-{name}"),
        "path": f"{kind}/{name}",
        "meaning": f"named missing {kind}",
    }


def pack_kwargs(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "repository_state_cid": cid_label("repo-state"),
        "task_id": TASK_ID,
        "target_source_cid": cid_label("target"),
        "surrounding_source_cid": cid_label("surround"),
        "test_source_cid": cid_label("test"),
        "scanned_tree_oid": FIXTURE_TREE,
        "source_tree_oid": FIXTURE_TREE,
        "objective_identity": OBJECTIVE_ID,
        "objective_revision": OBJECTIVE_REVISION,
        "policy_identity": cid_label("aseh-035-policy"),
        "freshness_bindings": freshness_bindings(),
        "invalidation": {
            "invalidation_triggers": ["tree-changed", "policy-changed"],
            "reusable_until_conditions": ["tree-unchanged"],
        },
        "identity_kind": "fixture",
        "evidence_kind": "fixture",
        "execution_mode": "simulated",
        "dependencies": [helper_dependency()],
    }
    fields.update(overrides)
    return fields


def build_minimal(**overrides: object) -> Any:
    return build_minimal_semantic_pack(**pack_kwargs(**overrides))


def build_bloated(**overrides: object) -> Any:
    extras = [unused_dependency(index) for index in range(16)]
    fields = pack_kwargs(**overrides)
    fields["dependencies"] = [helper_dependency(), *extras]
    return build_minimal_semantic_pack(**fields)


def generate_fixture_recipes() -> tuple[dict[str, Any], ...]:
    """Compact combinatorial recipes; one sealed fixture per required case."""

    recipes: list[dict[str, Any]] = []
    recipes.append(
        admit_recipe(
            {
                "fixture_id": "aseh-cp-reuse",
                "scenario": "eligible_reuse",
                "named_missing_kind": "symbol",
                "named_missing_name": "helper",
            }
        )
    )
    recipes.append(
        admit_recipe(
            {
                "fixture_id": "aseh-cp-tokens",
                "scenario": "token_before_after",
                "named_missing_kind": "symbol",
                "named_missing_name": "helper",
            }
        )
    )
    for kind in NAMED_MISSING_KINDS:
        recipes.append(
            admit_recipe(
                {
                    "fixture_id": f"aseh-cp-expand-{kind}",
                    "scenario": "expansion_precision_recall",
                    "named_missing_kind": kind,
                    "named_missing_name": f"named-{kind}",
                }
            )
        )
    recipes.append(
        admit_recipe(
            {
                "fixture_id": "aseh-cp-omission",
                "scenario": "seeded_critical_omission",
                "named_missing_kind": "symbol",
                "named_missing_name": "helper",
                "seeded_critical_omission": True,
                "critical_dependency": "omitted-critical",
            }
        )
    )
    for field in STALE_FIELDS:
        recipes.append(
            admit_recipe(
                {
                    "fixture_id": f"aseh-cp-stale-{field}",
                    "scenario": "stale_pack",
                    "stale_field": field,
                    "seeded_stale": True,
                    "named_missing_kind": "symbol",
                    "named_missing_name": "helper",
                }
            )
        )
    recipes.append(
        admit_recipe(
            {
                "fixture_id": "aseh-cp-fixture-as-live",
                "scenario": "fixture_as_live",
                "masquerade_live": True,
                "named_missing_kind": "symbol",
                "named_missing_name": "helper",
            }
        )
    )
    recipes.append(
        admit_recipe(
            {
                "fixture_id": "aseh-cp-unavailable-charge",
                "scenario": "unavailable_provider_charge",
                "named_missing_kind": "symbol",
                "named_missing_name": "helper",
            }
        )
    )
    return tuple(recipes)


def admit_recipe(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(payload, name="recipe")
    fixture_id = _text(data.get("fixture_id"), name="fixture_id", maximum=64)
    scenario = _text(data.get("scenario"), name="scenario", maximum=64)
    if data.get("live") not in {None, False}:
        raise FixtureAsLiveError(f"{fixture_id}: hermetic fixtures cannot be live")
    recipe = {
        "critical_dependency": _text(
            data.get("critical_dependency") or "helper",
            name="critical_dependency",
            maximum=64,
        ),
        "fixture_id": fixture_id,
        "live": False,
        "masquerade_live": _bool(
            data.get("masquerade_live") or False, name="masquerade_live"
        ),
        "named_missing_kind": _text(
            data.get("named_missing_kind") or "symbol",
            name="named_missing_kind",
            maximum=32,
        ),
        "named_missing_name": _text(
            data.get("named_missing_name") or "helper",
            name="named_missing_name",
            maximum=64,
        ),
        "population_kind": "hermetic_development",
        "scenario": scenario,
        "seeded_critical_omission": _bool(
            data.get("seeded_critical_omission") or False,
            name="seeded_critical_omission",
        ),
        "seeded_stale": _bool(data.get("seeded_stale") or False, name="seeded_stale"),
        "stale_field": None
        if data.get("stale_field") in {None, ""}
        else _text(data.get("stale_field"), name="stale_field", maximum=32),
    }
    if recipe["named_missing_kind"] not in NAMED_MISSING_KINDS:
        raise ContextPackBenchmarkError(
            f"{fixture_id}: named_missing_kind is not a closed value"
        )
    if recipe["stale_field"] is not None and recipe["stale_field"] not in STALE_FIELDS:
        raise ContextPackBenchmarkError(f"{fixture_id}: stale_field is not closed")
    encoded = compact_json(recipe)
    if len(encoded.encode("utf-8")) > MAX_SERIALIZED_RECIPE_BYTES:
        raise ContextPackBenchmarkError(
            f"{fixture_id}: recipe exceeds the serialized bound"
        )
    return canonical_object(recipe)


def recipe_identity(recipe: Mapping[str, Any]) -> str:
    return content_identity(admit_recipe(recipe))


def shared_equal_controls(*, task_inputs: str) -> dict[str, Any]:
    controls = {
        "acceptance_tests": content_identity(ACCEPTANCE_TESTS),
        "available_providers_and_models": content_identity(
            {"name": "providers", "status": "companion_present"}
        ),
        "human_intervention_policy": content_identity(HUMAN_INTERVENTION_POLICY),
        "maximum_retries": 3,
        "objective": OBJECTIVE_ID,
        "price_accounting": content_identity(
            {"name": "price", "status": "unavailable_unit_prices"}
        ),
        "repository_revision": REPOSITORY_TREE,
        "resource_limits": content_identity(RESOURCE_LIMITS),
        "task_inputs": _text(task_inputs, name="task_inputs", maximum=128),
    }
    if set(controls) != set(EQUAL_CONTROL_FIELDS):
        raise ContextPackBenchmarkError("equal controls drifted from the closed field set")
    return controls


def require_equal_controls(arm_controls: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    controls = _mapping(arm_controls, name="arm_controls")
    missing_arms = [arm for arm in PAIRED_ARMS if arm not in controls]
    extra_arms = sorted(set(controls) - set(PAIRED_ARMS))
    if missing_arms or extra_arms:
        raise ContextPackBenchmarkError(
            "pairing requires the three closed arms; "
            f"missing={missing_arms or 'none'} extra={extra_arms or 'none'}"
        )
    admitted: dict[str, dict[str, Any]] = {}
    for arm_id in PAIRED_ARMS:
        bundle = _mapping(controls[arm_id], name=f"{arm_id}.controls")
        unknown = sorted(set(bundle) - set(EQUAL_CONTROL_FIELDS))
        missing = [field for field in EQUAL_CONTROL_FIELDS if field not in bundle]
        if unknown or missing:
            raise ContextPackBenchmarkError(
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
        raise ContextPackBenchmarkError(
            "pairing rejects unequal controls: " + ", ".join(mismatches)
        )
    return dict(reference)


def controls_for_arms(controls: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    shared = {field: controls[field] for field in EQUAL_CONTROL_FIELDS}
    return {arm_id: dict(shared) for arm_id in PAIRED_ARMS}


def _stale_overrides(field: str) -> dict[str, object]:
    if field == "tree":
        return {"scanned_tree_oid": STALE_TREE, "source_tree_oid": STALE_TREE}
    if field == "objective":
        return {"objective_revision": "stale-objective-revision"}
    if field == "policy":
        return {"policy_identity": cid_label("stale-policy")}
    if field == "interface":
        return {
            "freshness_bindings": freshness_bindings(
                schema_identities=["stale.interface@1"]
            )
        }
    if field == "toolchain":
        return {
            "freshness_bindings": freshness_bindings(toolchain_identities=["pypy"])
        }
    if field == "environment":
        return {
            "freshness_bindings": freshness_bindings(
                environment_requirements=["PATH=/tmp/user-writable"]
            )
        }
    raise ContextPackBenchmarkError(f"unsupported stale field {field}")


def _store_pack(store: Any, record: Any, *, current: bool = False) -> tuple[Any, bytes]:
    data = encode_context_pack_envelope(record.to_dict())
    reference = store.put_candidate(data, cache_key=f"pack:{record.pack_cid}")
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


def _select_pack(store: Any, current: CurrentPackIdentity) -> tuple[Any, Any]:
    try:
        selection = select_current_minimal_pack(store, current)
        return selection, None
    except StaleIdentityError as exc:
        recorded = select_current_minimal_pack(
            store, current, require_selection=False
        )
        return recorded, exc


def _audit_units(*payloads: Mapping[str, Any] | bytes) -> int:
    total = 8 * len(EXACT_FRESHNESS_FIELDS)
    for payload in payloads:
        if isinstance(payload, bytes):
            total += len(payload)
        else:
            total += len(compact_json(payload).encode("utf-8"))
    return max(AUDIT_OVERHEAD_FLOOR, total)


def _require_audit(node: Mapping[str, Any], *, fixture_id: str) -> int:
    audit = _mapping(node, name=f"{fixture_id}.audit_overhead")
    if audit.get("truth_state") != "measured":
        raise MissingAuditCostError(f"{fixture_id}: missing audit cost")
    if "unit" not in audit or "sensor_id" not in audit:
        raise MissingAuditCostError(f"{fixture_id}: missing audit cost")
    try:
        value = _int(audit.get("value"), name=f"{fixture_id}.audit_overhead.value")
    except ContextPackBenchmarkError as exc:
        raise MissingAuditCostError(f"{fixture_id}: missing audit cost") from exc
    if value < AUDIT_OVERHEAD_FLOOR:
        raise MissingAuditCostError(f"{fixture_id}: missing audit cost")
    return value


def _unavailable_metric(reason: str) -> dict[str, str]:
    return unavailable(reason)


def _reuse_metric(*, reused: bool, live_reuse_granted: bool = False) -> dict[str, Any]:
    if live_reuse_granted:
        raise FixtureAsLiveError("hermetic results cannot grant live reuse")
    payload = measured_count(1 if reused else 0)
    payload["reused"] = reused
    payload["live_reuse_granted"] = False
    payload["eligible"] = reused
    return payload


def _omission_metric(
    *,
    seeded: bool,
    detected: bool,
    accepted: bool,
) -> dict[str, Any]:
    if accepted:
        raise AcceptedCriticalOmissionError("seeded critical omissions cannot be accepted")
    if seeded and not detected:
        raise AcceptedCriticalOmissionError(
            "seeded critical omissions must be detected and rejected"
        )
    payload = measured_count(1 if detected else 0)
    payload["seeded"] = seeded
    payload["detected"] = detected
    payload["accepted"] = False
    return payload


def _stale_metric(
    *,
    seeded: bool,
    rejected: bool,
    admitted: bool,
    stale_fields: Sequence[str] = (),
    masquerade_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    if seeded and admitted:
        raise StaleIdentityAdmissionError("seeded stale packs cannot be admitted")
    if seeded and not rejected:
        raise StaleIdentityAdmissionError("seeded stale packs must be rejected")
    payload = measured_count(1 if rejected else 0)
    payload["seeded"] = seeded
    payload["rejected"] = rejected
    payload["admitted"] = admitted
    payload["stale_fields"] = list(stale_fields)
    payload["masquerade_reasons"] = list(masquerade_reasons)
    return payload


def _net_provider_effect(token_delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "includes_audit_in_net_effect": True,
        "provider_reported_charge": _unavailable_metric("not_reported"),
        "token_based_estimated_charge": _unavailable_metric("not_reported"),
        "token_delta": token_delta,
        "unit_prices": _unavailable_metric("not_reported"),
    }


def _token_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if before.get("truth_state") != "measured" or after.get("truth_state") != "measured":
        return _unavailable_metric("not_admitted")
    before_value = _int(before.get("value"), name="context_tokens_before.value", minimum=1)
    after_value = _int(after.get("value"), name="context_tokens_after.value", minimum=1)
    delta = before_value - after_value
    if delta < 0:
        delta = 0
    return measured_tokens(delta)


def _arm_observation(
    *,
    arm_id: str,
    recipe: Mapping[str, Any],
    metrics: Mapping[str, Any],
    admitted: bool,
    disposition: str,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ContextPackBenchmarkError(
            f"{recipe['fixture_id']}: missing required metrics {missing}"
        )
    observation = {
        "admitted": admitted,
        "arm_id": arm_id,
        "disposition": disposition,
        "fixture_id": recipe["fixture_id"],
        "live": False,
        "live_reuse_granted": False,
        "metrics": canonical_object(metrics),
        "population_kind": "hermetic_development",
        "scenario": recipe["scenario"],
        "seeded_critical_omission": recipe["seeded_critical_omission"],
        "seeded_stale": recipe["seeded_stale"] or recipe["masquerade_live"],
        "simulated": True,
        "task_class": "context_pack_build",
        "truth_state": "measured",
    }
    return canonical_object(observation)


def measure_fixture(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Run current-tree Datasets/Accelerate/Kit measures for one recipe."""

    admitted_recipe = admit_recipe(recipe)
    fixture_id = admitted_recipe["fixture_id"]
    scenario = admitted_recipe["scenario"]
    kind = admitted_recipe["named_missing_kind"]
    name = admitted_recipe["named_missing_name"]
    parent = build_minimal()
    bloated = build_bloated()
    parent_bytes = encode_context_pack_envelope(parent.to_dict())
    bloated_bytes = encode_context_pack_envelope(bloated.to_dict())
    before_tokens = estimate_tokens(bloated_bytes)
    parent_tokens = estimate_tokens(parent_bytes)
    build_units = len(parent_bytes) + len(bloated_bytes)
    named_token = f"{kind}:{name}"
    catalog = [
        named_catalog_record(kind, name),
        named_catalog_record("symbol", "unused-whole-repo"),
    ]

    after_tokens: dict[str, Any]
    expansion_precision: dict[str, Any]
    expansion_recall: dict[str, Any]
    retrieval: dict[str, Any]
    reuse: dict[str, Any]
    omission: dict[str, Any]
    stale: dict[str, Any]
    admitted = False
    disposition = "rejected"
    candidate_tokens = parent_tokens
    retrieved_count = 0
    relevant_retrieved = 0
    relevant_count = 0
    audit_payloads: list[Mapping[str, Any] | bytes] = [parent_bytes, bloated_bytes]

    if scenario == "seeded_critical_omission":
        try:
            expand_incremental_pack(
                parent=parent,
                scanned_tree_oid=FIXTURE_TREE,
                named_missing=[named_token],
                catalog=catalog,
                critical_dependencies=[admitted_recipe["critical_dependency"]],
            )
            raise AcceptedCriticalOmissionError(
                f"{fixture_id}: seeded critical omission was not rejected"
            )
        except CriticalOmissionError:
            pass
        after_tokens = _unavailable_metric("not_admitted")
        expansion_precision = _unavailable_metric("not_admitted")
        expansion_recall = _unavailable_metric("not_admitted")
        retrieval = _unavailable_metric("not_admitted")
        reuse = _reuse_metric(reused=False)
        omission = _omission_metric(seeded=True, detected=True, accepted=False)
        stale = _unavailable_metric("not_applicable")
        disposition = "rejected_critical_omission"
    elif scenario == "expansion_precision_recall":
        result = expand_incremental_pack(
            parent=parent,
            scanned_tree_oid=FIXTURE_TREE,
            named_missing=[named_token],
            catalog=catalog,
            critical_dependencies=[]
            if kind != "symbol"
            else [name],
        )
        relevant_retrieved, retrieved_count, relevant_count = expansion_precision_recall(
            result.retrieved, result.relevant
        )
        expanded_bytes = encode_context_pack_envelope(result.pack.to_dict())
        candidate_tokens = estimate_tokens(expanded_bytes)
        after_tokens = measured_tokens(candidate_tokens)
        expansion_precision = measured_ratio(
            ratio_millionths(relevant_retrieved, retrieved_count)
        )
        expansion_recall = measured_ratio(
            ratio_millionths(relevant_retrieved, relevant_count)
        )
        retrieval = measured_compute(max(1, retrieved_count * 64 + len(expanded_bytes)))
        reuse = _reuse_metric(reused=False)
        omission = _omission_metric(seeded=False, detected=False, accepted=False)
        stale = _stale_metric(seeded=False, rejected=False, admitted=True)
        admitted = True
        disposition = "expanded"
        audit_payloads.append(expanded_bytes)
        build_units += len(expanded_bytes)
    elif scenario in {"eligible_reuse", "token_before_after", "unavailable_provider_charge"}:
        current = CurrentPackIdentity.from_envelope(parent.to_dict())
        reuse_current_root = scenario == "eligible_reuse"
        with tempfile.TemporaryDirectory(prefix="aseh-035-") as tmp:
            store = open_context_pack_store(tmp)
            try:
                if reuse_current_root:
                    _store_pack(store, bloated)
                    _store_pack(store, parent, current=True)
                else:
                    _store_pack(store, bloated, current=True)
                    _store_pack(store, parent)
                selection, error = _select_pack(store, current)
                if error is not None or selection.selected is None:
                    raise ContextPackBenchmarkError(
                        f"{fixture_id}: current-tree pack should have been admitted"
                    )
                reused = bool(selection.admission.reused)
                if reuse_current_root and not reused:
                    raise ContextPackBenchmarkError(
                        f"{fixture_id}: eligible current pack was not reused"
                    )
                if not reuse_current_root and reused:
                    raise ContextPackBenchmarkError(
                        f"{fixture_id}: non-current selection was recorded as reuse"
                    )
                candidate_tokens = estimate_tokens(
                    encode_context_pack_envelope(selection.selected.envelope)
                )
                if candidate_tokens >= before_tokens:
                    raise ContextPackBenchmarkError(
                        f"{fixture_id}: context_tokens_after must be below context_tokens_before"
                    )
                after_tokens = measured_tokens(candidate_tokens)
                expansion_precision = _unavailable_metric("not_applicable")
                expansion_recall = _unavailable_metric("not_applicable")
                retrieval = _unavailable_metric("not_applicable")
                reuse = _reuse_metric(reused=reused)
                omission = _omission_metric(seeded=False, detected=False, accepted=False)
                stale = _stale_metric(seeded=False, rejected=False, admitted=True)
                admitted = True
                disposition = "reuse" if reused else "selected"
                audit_payloads.append(selection.admission.to_dict())
            finally:
                store.close()
    elif scenario == "stale_pack":
        field = admitted_recipe["stale_field"]
        if field is None:
            raise ContextPackBenchmarkError(f"{fixture_id}: stale_field is required")
        fresh = parent
        stale_pack = build_minimal(**_stale_overrides(field))
        current = CurrentPackIdentity.from_envelope(fresh.to_dict())
        with tempfile.TemporaryDirectory(prefix="aseh-035-") as tmp:
            store = open_context_pack_store(tmp)
            try:
                _store_pack(store, stale_pack, current=True)
                selection, error = _select_pack(store, current)
                if error is None or selection.selected is not None:
                    raise StaleIdentityAdmissionError(
                        f"{fixture_id}: seeded stale pack was admitted"
                    )
                stale_fields = list(selection.admission.stale_fields)
                if field not in stale_fields:
                    stale_fields.append(field)
                after_tokens = _unavailable_metric("not_admitted")
                expansion_precision = _unavailable_metric("not_applicable")
                expansion_recall = _unavailable_metric("not_applicable")
                retrieval = _unavailable_metric("not_applicable")
                reuse = _reuse_metric(reused=False)
                omission = _unavailable_metric("not_applicable")
                stale = _stale_metric(
                    seeded=True,
                    rejected=True,
                    admitted=False,
                    stale_fields=stale_fields,
                )
                disposition = "rejected_stale"
                audit_payloads.append(selection.admission.to_dict())
                stale_bytes = encode_context_pack_envelope(stale_pack.to_dict())
                audit_payloads.append(stale_bytes)
                build_units += len(stale_bytes)
            finally:
                store.close()
    elif scenario == "fixture_as_live":
        fixture_pack = parent
        live_pack = build_minimal(
            identity_kind="live",
            evidence_kind="real",
            execution_mode="live",
        )
        current = CurrentPackIdentity.from_envelope(live_pack.to_dict())
        with tempfile.TemporaryDirectory(prefix="aseh-035-") as tmp:
            store = open_context_pack_store(tmp)
            try:
                _store_pack(store, fixture_pack, current=True)
                selection, error = _select_pack(store, current)
                if error is None or selection.selected is not None:
                    raise FixtureAsLiveError(
                        f"{fixture_id}: fixture pack masqueraded as live"
                    )
                reasons: list[str] = []
                for item in selection.invalidated:
                    reasons.extend(item.masquerade_reasons)
                if "fixture_as_live" not in reasons:
                    reasons.append("fixture_as_live")
                after_tokens = _unavailable_metric("not_admitted")
                expansion_precision = _unavailable_metric("not_applicable")
                expansion_recall = _unavailable_metric("not_applicable")
                retrieval = _unavailable_metric("not_applicable")
                reuse = _reuse_metric(reused=False)
                omission = _unavailable_metric("not_applicable")
                stale = _stale_metric(
                    seeded=True,
                    rejected=True,
                    admitted=False,
                    masquerade_reasons=reasons,
                )
                disposition = "rejected_fixture_as_live"
                audit_payloads.append(selection.admission.to_dict())
            finally:
                store.close()
    else:
        raise ContextPackBenchmarkError(f"{fixture_id}: unknown scenario {scenario}")

    before = measured_tokens(before_tokens)
    audit = measured_compute(_audit_units(*audit_payloads))
    build_cost = measured_compute(max(1, build_units))
    net = _net_provider_effect(_token_delta(before, after_tokens))
    metrics = {
        "audit_overhead": audit,
        "build_cost": build_cost,
        "context_tokens_after": after_tokens,
        "context_tokens_before": before,
        "critical_omission_detection": omission,
        "eligible_reuse": reuse,
        "expansion_precision": expansion_precision,
        "expansion_recall": expansion_recall,
        "net_provider_cost_effect": net,
        "retrieval_cost": retrieval,
        "stale_pack_rejection": stale,
    }
    direct_metrics = dict(metrics)
    if admitted:
        direct_after = measured_tokens(before_tokens)
        direct_metrics = dict(metrics)
        direct_metrics["context_tokens_after"] = direct_after
        direct_metrics["eligible_reuse"] = _reuse_metric(reused=False)
        direct_metrics["net_provider_cost_effect"] = _net_provider_effect(
            _token_delta(before, direct_after)
        )
        sealed_metrics = dict(metrics)
        if scenario == "expansion_precision_recall":
            sealed_after = measured_tokens(parent_tokens)
            sealed_metrics["context_tokens_after"] = sealed_after
            sealed_metrics["expansion_precision"] = _unavailable_metric("not_applicable")
            sealed_metrics["expansion_recall"] = _unavailable_metric("not_applicable")
            sealed_metrics["retrieval_cost"] = _unavailable_metric("not_applicable")
            sealed_metrics["eligible_reuse"] = _reuse_metric(reused=True)
            sealed_metrics["net_provider_cost_effect"] = _net_provider_effect(
                _token_delta(before, sealed_after)
            )
        else:
            sealed_metrics = dict(metrics)
        candidate_metrics = dict(metrics)
    else:
        direct_metrics = dict(metrics)
        sealed_metrics = dict(metrics)
        candidate_metrics = dict(metrics)

    observations = {
        "direct_minimal_orchestration_baseline": _arm_observation(
            arm_id="direct_minimal_orchestration_baseline",
            recipe=admitted_recipe,
            metrics=direct_metrics,
            admitted=admitted,
            disposition="direct_whole_repository" if admitted else disposition,
        ),
        "sealed_current_supervisor_baseline": _arm_observation(
            arm_id="sealed_current_supervisor_baseline",
            recipe=admitted_recipe,
            metrics=sealed_metrics,
            admitted=admitted,
            disposition="reuse"
            if admitted and sealed_metrics["eligible_reuse"].get("reused") is True
            else ("selected" if admitted else disposition),
        ),
        "candidate_optimized_supervisor": _arm_observation(
            arm_id="candidate_optimized_supervisor",
            recipe=admitted_recipe,
            metrics=candidate_metrics,
            admitted=admitted,
            disposition=disposition if admitted else disposition,
        ),
    }
    return {
        "admitted": admitted,
        "disposition": disposition,
        "fixture_id": fixture_id,
        "identity": recipe_identity(admitted_recipe),
        "live": False,
        "observations": observations,
        "recipe": admitted_recipe,
        "scenario": scenario,
        "seeded_critical_omission": admitted_recipe["seeded_critical_omission"],
        "seeded_stale": admitted_recipe["seeded_stale"]
        or admitted_recipe["masquerade_live"],
        "task_class": "context_pack_build",
    }


def _assert_unavailable_retained(payload: Mapping[str, Any], *, fixture_id: str) -> None:
    for path, node, state in _iter_evidence(payload):
        if state != "unavailable":
            continue
        if node.get("value", "unavailable") == 0 or node.get("count", "unavailable") == 0:
            raise ContextPackBenchmarkError(
                f"{fixture_id}.{path}: unavailable evidence cannot encode numeric zero"
            )
        for key, child in node.items():
            if type(child) is int and child == 0:
                raise ContextPackBenchmarkError(
                    f"{fixture_id}.{path}.{key}: unavailable evidence cannot encode numeric zero"
                )
        for forbidden in ("value", "count", "unit", "sensor_id", "estimator_id"):
            if forbidden in node:
                raise ContextPackBenchmarkError(
                    f"{fixture_id}.{path}: unavailable evidence cannot encode {forbidden}"
                )


def _iter_evidence(value: Any, *, path: str = ""):
    if isinstance(value, Mapping):
        state = value.get("truth_state")
        if isinstance(state, str):
            yield path, value, state
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_evidence(child, path=child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from _iter_evidence(child, path=child_path)


def admit_arm_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(observation, name="observation")
    fixture_id = _text(data.get("fixture_id"), name="fixture_id", maximum=64)
    if data.get("live") is True:
        raise FixtureAsLiveError(f"{fixture_id}: cannot mark an arm live")
    if data.get("live_reuse_granted") is True:
        raise FixtureAsLiveError(f"{fixture_id}: hermetic results cannot grant live reuse")
    metrics = _mapping(data.get("metrics"), name=f"{fixture_id}.metrics")
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ContextPackBenchmarkError(f"{fixture_id}: missing required metrics {missing}")
    _require_audit(metrics["audit_overhead"], fixture_id=fixture_id)
    omission = _mapping(
        metrics["critical_omission_detection"],
        name=f"{fixture_id}.critical_omission_detection",
    )
    if omission.get("truth_state") == "measured" and omission.get("accepted") is True:
        raise AcceptedCriticalOmissionError(
            f"{fixture_id}: accepted critical omission"
        )
    if (
        data.get("seeded_critical_omission") is True
        and omission.get("detected") is not True
    ):
        raise AcceptedCriticalOmissionError(
            f"{fixture_id}: seeded critical omission was not detected"
        )
    stale = metrics["stale_pack_rejection"]
    if isinstance(stale, Mapping) and stale.get("truth_state") == "measured":
        if data.get("seeded_stale") is True and stale.get("admitted") is True:
            raise StaleIdentityAdmissionError(
                f"{fixture_id}: stale identity admission"
            )
        if data.get("seeded_stale") is True and stale.get("rejected") is not True:
            raise StaleIdentityAdmissionError(
                f"{fixture_id}: seeded stale pack was not rejected"
            )
    _assert_unavailable_retained(metrics, fixture_id=fixture_id)
    return canonical_object(data)


def pair_fixture_observations(
    observations: Mapping[str, Mapping[str, Any]],
    *,
    controls: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    arms = _mapping(observations, name="observations")
    missing = [arm for arm in PAIRED_ARMS if arm not in arms]
    if missing:
        raise ContextPackBenchmarkError(f"missing paired inputs for arms: {missing}")
    admitted_obs = {
        arm_id: admit_arm_observation(arms[arm_id]) for arm_id in PAIRED_ARMS
    }
    fixture_ids = {admitted_obs[arm]["fixture_id"] for arm in PAIRED_ARMS}
    if len(fixture_ids) != 1:
        raise ContextPackBenchmarkError("paired inputs must share one fixture_id")
    if any(admitted_obs[arm].get("live") is True for arm in PAIRED_ARMS):
        raise FixtureAsLiveError("hermetic pairing cannot mark an arm live")
    return {
        "admitted": admitted_obs["candidate_optimized_supervisor"]["admitted"],
        "controls": {field: controls[field] for field in EQUAL_CONTROL_FIELDS},
        "fixture_id": next(iter(fixture_ids)),
        "identity": recipe_identity(recipe),
        "live": False,
        "observations": admitted_obs,
        "recipe": admit_recipe(recipe),
        "scenario": recipe["scenario"],
        "seeded_critical_omission": recipe["seeded_critical_omission"],
        "seeded_stale": recipe["seeded_stale"] or recipe["masquerade_live"],
        "task_class": "context_pack_build",
    }


def admit_paired_fixture(pair: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(pair, name="paired_fixture")
    if "observations" not in data:
        raise AggregateOnlyError("paired fixtures are required; aggregate-only results are invalid")
    recipe = admit_recipe(data.get("recipe") or {"fixture_id": data.get("fixture_id"), "scenario": data.get("scenario")})
    controls = _mapping(data.get("controls"), name="controls")
    return pair_fixture_observations(
        _mapping(data.get("observations"), name="observations"),
        controls=controls,
        recipe=recipe,
    )


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


def compute_campaign_statistics(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise ContextPackBenchmarkError("paired statistics are unavailable for an empty population")
    token_deltas: list[int] = []
    audit_values: list[int] = []
    reuse_count = 0
    omission_detected = 0
    omission_accepted = 0
    stale_rejected = 0
    stale_admitted = 0
    precision_values: list[int] = []
    recall_values: list[int] = []
    for pair in pairs:
        candidate = _mapping(
            pair["observations"]["candidate_optimized_supervisor"],
            name="candidate",
        )
        metrics = _mapping(candidate["metrics"], name="candidate.metrics")
        audit_values.append(_require_audit(metrics["audit_overhead"], fixture_id=pair["fixture_id"]))
        reuse = metrics["eligible_reuse"]
        if reuse.get("truth_state") == "measured" and reuse.get("reused") is True:
            reuse_count += 1
        omission = metrics["critical_omission_detection"]
        if omission.get("truth_state") == "measured":
            if omission.get("detected") is True:
                omission_detected += 1
            if omission.get("accepted") is True:
                omission_accepted += 1
        stale = metrics["stale_pack_rejection"]
        if isinstance(stale, Mapping) and stale.get("truth_state") == "measured":
            if stale.get("rejected") is True:
                stale_rejected += 1
            if stale.get("admitted") is True and pair.get("seeded_stale") is True:
                stale_admitted += 1
        precision = metrics["expansion_precision"]
        if precision.get("truth_state") == "measured":
            precision_values.append(_int(precision["value"], name="expansion_precision"))
        recall = metrics["expansion_recall"]
        if recall.get("truth_state") == "measured":
            recall_values.append(_int(recall["value"], name="expansion_recall"))
        net = _mapping(metrics["net_provider_cost_effect"], name="net_provider_cost_effect")
        token_delta = net["token_delta"]
        if token_delta.get("truth_state") == "measured":
            token_deltas.append(_int(token_delta["value"], name="token_delta"))
    if omission_accepted:
        raise AcceptedCriticalOmissionError("campaign accepted a critical omission")
    if stale_admitted:
        raise StaleIdentityAdmissionError("campaign admitted a stale pack")
    return {
        "audit_overhead": {
            "includes_audit_in_net_effect": True,
            "median_compute_units": _median_int(audit_values),
            "total_compute_units": sum(audit_values),
            "unit": "compute_units",
        },
        "critical_omission_detection": {
            "accepted": 0,
            "detected": omission_detected,
            "seeded": sum(1 for item in pairs if item.get("seeded_critical_omission") is True),
        },
        "eligible_reuse": {
            "count": reuse_count,
            "total": len(pairs),
        },
        "expansion_precision": measured_ratio(_median_int(precision_values))
        if precision_values
        else _unavailable_metric("not_applicable"),
        "expansion_recall": measured_ratio(_median_int(recall_values))
        if recall_values
        else _unavailable_metric("not_applicable"),
        "net_provider_cost_effect": _net_provider_effect(
            measured_tokens(_median_int(token_deltas))
            if token_deltas
            else _unavailable_metric("not_admitted")
        ),
        "pair_count": len(pairs),
        "stale_pack_rejection": {
            "admitted": 0,
            "rejected": stale_rejected,
            "seeded": sum(1 for item in pairs if item.get("seeded_stale") is True),
        },
    }


def run_paired_campaign(
    recipes: Sequence[Mapping[str, Any]] | None = None,
    *,
    controls: Mapping[str, Any] | None = None,
    arm_controls: Mapping[str, Mapping[str, Any]] | None = None,
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
    paired = []
    for recipe in catalogue:
        measured = measure_fixture(recipe)
        pair = pair_fixture_observations(
            measured["observations"],
            controls=shared,
            recipe=recipe,
        )
        paired.append(pair)
    if not paired:
        raise AggregateOnlyError("paired fixtures are required")
    statistics = compute_campaign_statistics(paired)
    audit_total = statistics["audit_overhead"]["total_compute_units"]
    result = {
        "arms": list(PAIRED_ARMS),
        "audit_overhead": statistics["audit_overhead"],
        "authority": False,
        "controls": shared,
        "critical_omissions_accepted": 0,
        "fixture_count": len(catalogue),
        "hermetic_sufficient_for_production_promotion": False,
        "live": False,
        "live_reuse_granted": False,
        "paired_fixtures": paired,
        "population_kind": "hermetic_development",
        "promotion_without_paired_campaign": False,
        "recipes": list(catalogue),
        "required_metrics": list(REQUIRED_METRICS),
        "stale_packs_admitted": 0,
        "statistics": statistics,
        "vectors_identity": vectors_identity,
    }
    if audit_total < AUDIT_OVERHEAD_FLOOR:
        raise MissingAuditCostError("campaign is missing audit cost")
    _reject_floats(result)
    _assert_unavailable_retained(result, fixture_id="campaign")
    return result


def admit_campaign(result: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(result, name="campaign")
    if data.get("live") is True:
        raise FixtureAsLiveError("campaign cannot be live")
    if data.get("live_reuse_granted") is True:
        raise FixtureAsLiveError("hermetic results cannot grant live reuse")
    if data.get("hermetic_sufficient_for_production_promotion") is not False:
        raise FixtureAsLiveError("hermetic evidence cannot satisfy production promotion")
    if data.get("authority") is not False:
        raise ContextPackBenchmarkError("benchmark evidence is not authority")
    pairs = data.get("paired_fixtures")
    if not isinstance(pairs, list) or not pairs:
        raise AggregateOnlyError(
            "aggregate-only results are invalid; paired fixtures are required"
        )
    admitted_pairs = [admit_paired_fixture(item) for item in pairs]
    if data.get("critical_omissions_accepted") not in {0, None}:
        raise AcceptedCriticalOmissionError("campaign accepted a critical omission")
    if data.get("stale_packs_admitted") not in {0, None}:
        raise StaleIdentityAdmissionError("campaign admitted a stale pack")
    audit = data.get("audit_overhead")
    if not isinstance(audit, Mapping) or "total_compute_units" not in audit:
        raise MissingAuditCostError("missing audit cost")
    try:
        audit_total = _int(
            audit.get("total_compute_units"),
            name="audit_overhead.total_compute_units",
        )
    except ContextPackBenchmarkError as exc:
        raise MissingAuditCostError("missing audit cost") from exc
    if audit_total < AUDIT_OVERHEAD_FLOOR:
        raise MissingAuditCostError("missing audit cost")
    payload = dict(data)
    payload["paired_fixtures"] = admitted_pairs
    return canonical_object(payload)


def build_context_pack_manifest(result: Mapping[str, Any]) -> dict[str, Any]:
    campaign = admit_campaign(result)
    fixtures = []
    for pair in campaign["paired_fixtures"]:
        candidate = pair["observations"]["candidate_optimized_supervisor"]
        fixtures.append(
            {
                "admitted": pair["admitted"],
                "disposition": candidate["disposition"],
                "fixture_id": pair["fixture_id"],
                "identity": pair["identity"],
                "live": False,
                "metrics": candidate["metrics"],
                "scenario": pair["scenario"],
                "seeded_critical_omission": pair["seeded_critical_omission"],
                "seeded_stale": pair["seeded_stale"],
            }
        )
    payload = {
        "arm_constraints": {
            "candidate_optimized_supervisor": list(CANDIDATE_ARM_CONSTRAINTS),
            "direct_minimal_orchestration_baseline": list(DIRECT_ARM_CONSTRAINTS),
            "sealed_current_supervisor_baseline": list(SEALED_CURRENT_ARM_CONSTRAINTS),
        },
        "arms": list(PAIRED_ARMS),
        "audit_overhead": campaign["audit_overhead"],
        "authority": False,
        "controls": campaign["controls"],
        "count": len(fixtures),
        "critical_omissions_accepted": 0,
        "equal_control_fields": list(EQUAL_CONTROL_FIELDS),
        "explicit_unavailable_fields": list(collect_unavailable_fields(campaign)),
        "fixtures": fixtures,
        "hermetic_sufficient_for_production_promotion": False,
        "interface": BENCHMARK_INTERFACE,
        "live": False,
        "live_reuse_granted": False,
        "objective_id": OBJECTIVE_ID,
        "objective_revision": OBJECTIVE_REVISION,
        "policy_identity": POLICY_IDENTITY,
        "population_kind": "hermetic_development",
        "program_id": PROGRAM_ID,
        "promotion_without_paired_campaign": False,
        "recipe_identity": campaign["vectors_identity"],
        "repository_commit": REPOSITORY_COMMIT,
        "repository_tree": REPOSITORY_TREE,
        "required_metrics": list(REQUIRED_METRICS),
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "stale_packs_admitted": 0,
        "statistics": campaign["statistics"],
        "status": "sealed",
        "task_id": TASK_ID,
        "vectors_identity": campaign["vectors_identity"],
    }
    body = {key: value for key, value in payload.items() if key not in {"identity", "manifest_cid"}}
    identity = content_identity(body)
    payload["identity"] = identity
    payload["manifest_cid"] = identity
    _reject_floats(payload)
    if payload["hermetic_sufficient_for_production_promotion"] is not False:
        raise FixtureAsLiveError("hermetic evidence cannot satisfy production promotion")
    if payload["live_reuse_granted"] is not False:
        raise FixtureAsLiveError("hermetic results cannot grant live reuse")
    return canonical_object(payload)


def seal_artifacts(
    recipes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    catalogue = tuple(admit_recipe(item) for item in (recipes or generate_fixture_recipes()))
    controls = shared_equal_controls(
        task_inputs=content_identity({"recipes": list(catalogue)})
    )
    campaign = run_paired_campaign(catalogue, controls=controls)
    manifest = build_context_pack_manifest(campaign)
    return {
        "campaign": campaign,
        "manifest": manifest,
        "recipes": list(catalogue),
    }


def write_sealed_artifacts(directory: Path | None = None) -> dict[str, Path]:
    target = directory or PACKAGE_DIR
    sealed = seal_artifacts()
    path = target / "context_pack_manifest.json"
    path.write_text(pretty_json(sealed["manifest"]), encoding="utf-8")
    return {"manifest": path}


def verify_sealed_artifacts(directory: Path | None = None) -> dict[str, Any]:
    target = directory or PACKAGE_DIR
    sealed = seal_artifacts()
    observed = (target / "context_pack_manifest.json").read_text(encoding="utf-8")
    expected = pretty_json(sealed["manifest"])
    if observed != expected:
        raise ContextPackBenchmarkError(
            "context_pack_manifest.json drifted from the sealed campaign"
        )
    payload = json.loads(observed)
    if payload["count"] != len(sealed["recipes"]):
        raise ContextPackBenchmarkError("sealed fixture count drifted")
    admit_campaign(sealed["campaign"])
    return {
        "fixture_count": payload["count"],
        "manifest_cid": payload["manifest_cid"],
        "unique_identities": len({item["identity"] for item in payload["fixtures"]}),
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContextPackBenchmarkError(f"{path.name} is unreadable") from exc
    return _mapping(payload, name=path.name)


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
                "critical_omissions_accepted": campaign["critical_omissions_accepted"],
                "stale_packs_admitted": campaign["stale_packs_admitted"],
                "required_metrics": campaign["required_metrics"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
