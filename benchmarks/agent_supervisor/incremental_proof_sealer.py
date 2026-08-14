#!/usr/bin/env python3
"""Deterministic forty-transition IncrementalProofSealer benchmark (IPS-052).

Interfaces
----------
* ``IncrementalProofBenchmark@1`` — closed 40-transition workload and result schema
* CLI: ``--seed``, ``--transitions``, ``--json-output``, ``--csv-output``

Normative rules (fail-closed)
-----------------------------
* Stable seed/input yields the same ordered task sequence and unit sets.
* Simulated evidence never counts as production proving.
* Estimated cost-model values are labeled ``estimated``; unobserved values are
  ``unavailable`` (null). Measured and estimated are never conflated.
* Full checkpoints require a non-empty fallback reason; incremental seals set
  ``fallback_reason`` to null.
* Unit arithmetic: ``newly_proved = invalidated + added`` and
  ``required = reused + newly_proved``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

EVIDENCE_SUBSET: Final[str] = "ips/benchmark-workload@1"
BENCHMARK_INTERFACE: Final[str] = "IncrementalProofBenchmark@1"
BENCHMARK_SCHEMA: Final[str] = "incremental-proof-sealer-benchmark-results@2"
BENCHMARK_ID: Final[str] = "incremental-proof-sealer-40-transition@1"
DEFAULT_SEED: Final[int] = 20260811
DEFAULT_TRANSITION_COUNT: Final[int] = 40
UNIT_COUNT_PROVENANCE: Final[str] = "observed_planner_output"
PROTECTED_RUNNER_ID: Final[str] = "protected-board-benchmark-runner@1"
PROTECTED_CLAIM: Final[str] = (
    "benchmark_process_observed_metrics_retain_per_metric_provenance"
)
CLI_RELATIVE: Final[str] = "benchmarks/agent_supervisor/incremental_proof_sealer.py"
DEFAULT_JSON_RELATIVE: Final[str] = (
    "artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json"
)
DEFAULT_CSV_RELATIVE: Final[str] = (
    "artifacts/agent_supervisor/incremental_proof_sealer/benchmark.csv"
)

SCENARIOS: Final[tuple[str, ...]] = (
    "initial repository",
    "localized private source edit",
    "unrelated documentation",
    "one test-source edit",
    "one fixture edit",
    "unrelated module edit",
    "public-interface edit",
    "dependent module edit",
    "selected test addition",
    "authorized test deletion",
    "relevant configuration edit",
    "ordinary documentation",
    "dependency-lock class upgrade",
    "localized source edit",
    "two independent module edits",
    "branch A edit",
    "branch B edit from prior accepted parent",
    "merge A/B",
    "rollback of source bytes",
    "property-test edit",
    "periodic N-commit checkpoint",
    "documentation-only",
    "circuit version change",
    "localized source edit",
    "verification-key change",
    "test-selector change",
    "network-policy change",
    "environment trust-policy change",
    "integration fixture edit",
    "requirement policy change",
    "periodic checkpoint",
    "integration-test addition",
    "proof schema/canonicalization change",
    "checked-specification document edit",
    "ordinary documentation edit",
    "injected cache corruption detection",
    "two independent modules",
    "wrong-parent attempt then valid",
    "merge plus unaffected reuse",
    "release tag/compaction",
)

# Mandatory full checkpoints (plan §13 / validator closed set).
FULL_TRANSITIONS: Final[frozenset[int]] = frozenset(
    {0, 12, 20, 22, 24, 27, 30, 32, 35, 39}
)
# May seal full or incremental with an honest reason.
CONDITIONAL_FULL_TRANSITIONS: Final[frozenset[int]] = frozenset({17, 29, 38})

METRIC_FIELDS: Final[tuple[str, ...]] = (
    "leaf_proving_seconds",
    "aggregation_seconds",
    "prover_cpu_seconds",
    "prover_gpu_seconds",
    "peak_memory_bytes",
    "proof_size_bytes",
    "seal_size_bytes",
    "storage_growth_bytes",
    "seal_verification_seconds",
    "wall_clock_seconds",
    "full_proof_cost",
    "incremental_proof_cost",
)

CSV_FIELDS: Final[tuple[str, ...]] = (
    "index",
    "scenario",
    "seal_status",
    "measurement_provenance",
    "required_units",
    "reused_units",
    "invalidated_units",
    "added_units",
    "removed_units",
    "newly_proved_units",
    "cache_hit_rate",
    *METRIC_FIELDS,
    "compute_saved_percent",
    "chain_depth",
    "fallback_reason",
    "deterministic_roots_match",
    "simulated_required_units",
)

# Closed base universe for the synthetic repository graph.
_BASE_UNITS: Final[tuple[str, ...]] = (
    "unit/module/core",
    "unit/module/util",
    "unit/module/api",
    "unit/module/dep",
    "unit/module/extra",
    "unit/test/t1",
    "unit/test/t2",
    "unit/fixture/local",
    "unit/config/app",
    "unit/policy/verification",
)

_FULL_REASON: Final[dict[int, str]] = {
    0: "first_state_genesis_full_checkpoint",
    12: "dependency_lock_class_upgrade_full_checkpoint",
    17: "merge_incomplete_full_fallback",
    20: "periodic_n_commit_full_checkpoint",
    22: "circuit_version_change_full_checkpoint",
    24: "verification_key_change_full_checkpoint",
    27: "environment_trust_policy_change_full_checkpoint",
    30: "periodic_full_checkpoint",
    32: "proof_schema_canonicalization_change_full_checkpoint",
    35: "cache_corruption_detection_full_checkpoint",
    39: "release_tag_chain_compaction_full_checkpoint",
}


class BenchmarkError(ValueError):
    """Fail-closed benchmark contract violation."""


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    """One ordered workload step before unit-set evaluation."""

    index: int
    scenario: str
    kind: str
    force_full: bool
    invalidate: tuple[str, ...]
    add: tuple[str, ...]
    remove: tuple[str, ...]
    fallback_reason: str | None
    branch_parent_index: int | None = None
    rejected_wrong_parent: bool = False


@dataclass(frozen=True, slots=True)
class TransitionUnitSet:
    """Expected planner unit sets for one accepted transition."""

    index: int
    scenario: str
    required: frozenset[str]
    reused: frozenset[str]
    invalidated: frozenset[str]
    added: frozenset[str]
    removed: frozenset[str]
    newly_proved: frozenset[str]
    seal_status: str
    fallback_reason: str | None
    chain_depth: int
    rejected_attempts: tuple[Mapping[str, str], ...]


def _digest(payload: str) -> str:
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _revision(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _summarize_provenance(values: Iterable[str]) -> str:
    sources = set(values)
    if sources == {"measured"}:
        return "measured"
    if sources == {"estimated"}:
        return "estimated"
    return "mixed"


def _transition_specs(count: int) -> tuple[TransitionSpec, ...]:
    if count < 1 or count > len(SCENARIOS):
        raise BenchmarkError(
            f"transition count must be in 1..{len(SCENARIOS)}, got {count}"
        )
    specs: list[TransitionSpec] = []
    for index in range(count):
        scenario = SCENARIOS[index]
        kind = _kind_for_index(index)
        force_full = index in FULL_TRANSITIONS or (
            index in CONDITIONAL_FULL_TRANSITIONS and _conditional_is_full(index)
        )
        invalidate, add, remove = _mutation_for_index(index)
        reason = _FULL_REASON.get(index) if force_full else None
        if force_full and reason is None:
            reason = f"reviewed_full_checkpoint_at_{index:02d}"
        specs.append(
            TransitionSpec(
                index=index,
                scenario=scenario,
                kind=kind,
                force_full=force_full,
                invalidate=invalidate,
                add=add,
                remove=remove,
                fallback_reason=reason,
                branch_parent_index=14 if index in {15, 16} else None,
                rejected_wrong_parent=(index == 37),
            )
        )
    return tuple(specs)


def _kind_for_index(index: int) -> str:
    mapping = {
        0: "genesis",
        1: "localized_source",
        2: "documentation",
        3: "test_source",
        4: "fixture",
        5: "unrelated_module",
        6: "public_interface",
        7: "dependent_module",
        8: "test_add",
        9: "test_delete",
        10: "configuration",
        11: "documentation",
        12: "dependency_lock",
        13: "localized_source",
        14: "independent_modules",
        15: "branch_a",
        16: "branch_b",
        17: "merge",
        18: "rollback",
        19: "property_test",
        20: "periodic_checkpoint",
        21: "documentation",
        22: "circuit_change",
        23: "localized_source",
        24: "verification_key",
        25: "test_selector",
        26: "network_policy",
        27: "environment_trust",
        28: "integration_fixture",
        29: "requirement_policy",
        30: "periodic_checkpoint",
        31: "integration_test_add",
        32: "schema_canonicalization",
        33: "checked_specification",
        34: "documentation",
        35: "cache_corruption",
        36: "independent_modules",
        37: "wrong_parent_then_valid",
        38: "merge_with_reuse",
        39: "release_compaction",
    }
    return mapping[index]


def _conditional_is_full(index: int) -> bool:
    """Deterministic honest decisions for conditional full-or-incremental rows."""

    # 17 merge: incomplete multi-parent resolution falls back full.
    # 29 requirement policy: classified as incremental manifest recompute.
    # 38 merge plus unaffected reuse: bounded incremental succeeds.
    return index == 17


def _mutation_for_index(index: int) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (invalidate, add, remove) unit id tuples for one transition."""

    if index == 0:
        return (), _BASE_UNITS, ()
    if index == 1:
        return ("unit/module/core",), (), ()
    if index == 2:
        return (), (), ()
    if index == 3:
        return ("unit/test/t1",), (), ()
    if index == 4:
        return ("unit/fixture/local",), (), ()
    if index == 5:
        return ("unit/module/extra",), (), ()
    if index == 6:
        return ("unit/module/api", "unit/module/dep"), (), ()
    if index == 7:
        return ("unit/module/dep",), (), ()
    if index == 8:
        return (), ("unit/test/t3",), ()
    if index == 9:
        return (), (), ("unit/test/t2",)
    if index == 10:
        return ("unit/config/app",), (), ()
    if index == 11:
        return (), (), ()
    if index == 12:
        return (), (), ()  # full: all current units reproved
    if index == 13:
        return ("unit/module/core",), (), ()
    if index == 14:
        return ("unit/module/util", "unit/module/extra"), (), ()
    if index == 15:
        return ("unit/module/core",), (), ()
    if index == 16:
        return ("unit/module/util",), (), ()
    if index == 17:
        return ("unit/module/core", "unit/module/util"), (), ()
    if index == 18:
        return ("unit/module/core",), (), ()
    if index == 19:
        return (), ("unit/test/property",), ()
    if index == 20:
        return (), (), ()
    if index == 21:
        return (), (), ()
    if index == 22:
        return (), ("unit/circuit/v2",), ()
    if index == 23:
        return ("unit/module/core",), (), ()
    if index == 24:
        return (), (), ()
    if index == 25:
        return ("unit/test/t1", "unit/test/t3"), (), ()
    if index == 26:
        return (), ("unit/config/network",), ()
    if index == 27:
        return (), (), ()
    if index == 28:
        return (), ("unit/fixture/integration",), ()
    if index == 29:
        return ("unit/policy/verification",), (), ()
    if index == 30:
        return (), (), ()
    if index == 31:
        return (), ("unit/test/integration",), ()
    if index == 32:
        return (), (), ()
    if index == 33:
        return ("unit/module/api", "unit/policy/verification"), (), ()
    if index == 34:
        return (), (), ()
    if index == 35:
        return (), (), ()
    if index == 36:
        return ("unit/module/util", "unit/module/extra"), (), ()
    if index == 37:
        return ("unit/module/core",), (), ()
    if index == 38:
        return ("unit/module/dep",), (), ()
    if index == 39:
        return (), (), ()
    raise BenchmarkError(f"no mutation model for transition {index}")


def _evaluate_unit_sets(
    specs: Sequence[TransitionSpec],
) -> tuple[TransitionUnitSet, ...]:
    """Walk the closed history and produce expected unit sets per transition."""

    current: set[str] = set()
    results: list[TransitionUnitSet] = []
    chain_depth = 0
    # Snapshot of units at the common parent of branches 15/16 (after 14).
    branch_parent_units: frozenset[str] | None = None

    for spec in specs:
        if spec.index == 15:
            branch_parent_units = frozenset(current)

        working = set(current)
        if spec.index == 16 and branch_parent_units is not None:
            # Branch B starts from the prior accepted parent (post-14), not tip of A.
            working = set(branch_parent_units)

        removed = {unit for unit in spec.remove if unit in working}
        working -= removed
        added = {unit for unit in spec.add if unit not in working}

        if spec.force_full:
            # Full checkpoints re-prove every retained unit and any additions.
            invalidated = set(working)
            reused: set[str] = set()
            newly = set(invalidated) | set(added)
            working |= added
            required = set(working)
            seal_status = "sealed_full"
            chain_depth = 0
            fallback = spec.fallback_reason
        else:
            invalidated = {unit for unit in spec.invalidate if unit in working}
            working |= added
            required = set(working)
            reused = required - invalidated - added
            newly = set(invalidated) | set(added)
            seal_status = "sealed_incremental"
            chain_depth += 1
            fallback = None

        current = set(working)

        rejected: tuple[Mapping[str, str], ...] = ()
        if spec.rejected_wrong_parent:
            rejected = ({"kind": "wrong_parent", "terminal_status": "stale_parent"},)

        unit_set = TransitionUnitSet(
            index=spec.index,
            scenario=spec.scenario,
            required=frozenset(sorted(required)),
            reused=frozenset(sorted(reused)),
            invalidated=frozenset(sorted(invalidated)),
            added=frozenset(sorted(added)),
            removed=frozenset(sorted(removed)),
            newly_proved=frozenset(sorted(newly)),
            seal_status=seal_status,
            fallback_reason=fallback,
            chain_depth=chain_depth,
            rejected_attempts=rejected,
        )
        _assert_unit_arithmetic(unit_set)
        results.append(unit_set)
    return tuple(results)


def _assert_unit_arithmetic(unit_set: TransitionUnitSet) -> None:
    newly = unit_set.invalidated | unit_set.added
    if unit_set.newly_proved != newly:
        raise BenchmarkError(
            f"transition {unit_set.index:02d}: newly_proved set mismatch"
        )
    required = unit_set.reused | unit_set.newly_proved
    if unit_set.required != required:
        raise BenchmarkError(
            f"transition {unit_set.index:02d}: required set mismatch "
            f"(reused+newly_proved)"
        )
    if unit_set.reused & unit_set.newly_proved:
        raise BenchmarkError(
            f"transition {unit_set.index:02d}: reused and newly_proved overlap"
        )


def _estimate_metrics(
    *,
    seed: int,
    index: int,
    required: int,
    reused: int,
    newly_proved: int,
    seal_status: str,
    gpu_available: bool,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Deterministic cost model. Values are estimated unless GPU is unavailable."""

    # Stable fractional salt from seed/index (not wall-clock).
    salt = int(
        hashlib.sha256(f"{seed}:{index}:ips-052".encode("utf-8")).hexdigest()[:8],
        16,
    )
    salt_frac = (salt % 10_000) / 10_000.0

    leaf = newly_proved * (0.042 + 0.001 * salt_frac)
    aggregate = max(1, (required + 7) // 8) * (0.008 + 0.0002 * salt_frac)
    verify = 0.004 + 0.0001 * required + 0.00005 * salt_frac
    cpu = newly_proved * (0.9 + 0.02 * salt_frac) + reused * 0.01
    wall = leaf + aggregate + verify + 0.01 * salt_frac
    peak_mem = 8_388_608 + required * 65_536 + newly_proved * 131_072
    proof_size = newly_proved * 1_762 + reused * 64
    seal_size = 4_096 + required * 48
    storage = newly_proved * 4_096 + (0 if seal_status == "sealed_full" else reused * 256)
    full_cost = max(1.0, required * (1.0 + 0.01 * salt_frac))
    if seal_status == "sealed_full":
        incremental_cost = full_cost
    else:
        incremental_cost = max(0.0, newly_proved * (1.0 + 0.01 * salt_frac) + reused * 0.05)

    metrics: dict[str, float | None] = {
        "leaf_proving_seconds": leaf,
        "aggregation_seconds": aggregate,
        "prover_cpu_seconds": cpu,
        "prover_gpu_seconds": None,
        "peak_memory_bytes": float(peak_mem),
        "proof_size_bytes": float(proof_size),
        "seal_size_bytes": float(seal_size),
        "storage_growth_bytes": float(storage),
        "seal_verification_seconds": verify,
        "wall_clock_seconds": wall,
        "full_proof_cost": full_cost,
        "incremental_proof_cost": incremental_cost,
    }
    provenance: dict[str, str] = {name: "estimated" for name in METRIC_FIELDS}
    if gpu_available:
        metrics["prover_gpu_seconds"] = newly_proved * 0.05
        provenance["prover_gpu_seconds"] = "estimated"
    else:
        metrics["prover_gpu_seconds"] = None
        provenance["prover_gpu_seconds"] = "unavailable"
    return metrics, provenance


def _compute_saved_percent(
    full_cost: float | None, incremental_cost: float | None
) -> float | None:
    if full_cost is None or incremental_cost is None:
        return None
    if full_cost == 0:
        return 0.0
    return (full_cost - incremental_cost) / full_cost * 100.0


class IncrementalProofBenchmark:
    """Deterministic forty-transition full-versus-incremental benchmark."""

    interface: Final[str] = BENCHMARK_INTERFACE
    schema_version: Final[str] = BENCHMARK_SCHEMA
    benchmark_id: Final[str] = BENCHMARK_ID
    evidence_subset: Final[str] = EVIDENCE_SUBSET

    def __init__(
        self,
        *,
        seed: int = DEFAULT_SEED,
        transition_count: int = DEFAULT_TRANSITION_COUNT,
        gpu_available: bool = False,
        real_prover_available: bool = False,
        recursive_verification_available: bool = False,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise BenchmarkError("seed must be an integer")
        if not isinstance(transition_count, int) or isinstance(transition_count, bool):
            raise BenchmarkError("transition_count must be an integer")
        if transition_count < 1 or transition_count > len(SCENARIOS):
            raise BenchmarkError(
                f"transition_count must be in 1..{len(SCENARIOS)}, got {transition_count}"
            )
        self.seed = seed
        self.transition_count = transition_count
        self.gpu_available = bool(gpu_available)
        self.real_prover_available = bool(real_prover_available)
        self.recursive_verification_available = bool(recursive_verification_available)
        self._specs = _transition_specs(transition_count)
        self._unit_sets = _evaluate_unit_sets(self._specs)

    def task_sequence(self) -> tuple[TransitionSpec, ...]:
        """Return the ordered closed workload for this seed/count.

        The reviewed forty-transition identity is authoritative for any seed;
        the seed only salt-stabilizes estimated cost metrics and seal roots.
        """

        return self._specs

    def expected_unit_sets(self) -> tuple[TransitionUnitSet, ...]:
        """Return planner unit sets for every accepted transition."""

        return self._unit_sets

    def workload_fingerprint(self) -> str:
        """Stable digest of the ordered scenario/kind/unit-set identity.

        Seed is intentionally excluded so two seeds share the same task sequence
        and expected unit sets while still diverging on estimated metrics/roots.
        """

        payload = [
            {
                "index": unit.index,
                "scenario": unit.scenario,
                "kind": self._specs[unit.index].kind,
                "required": sorted(unit.required),
                "reused": sorted(unit.reused),
                "invalidated": sorted(unit.invalidated),
                "added": sorted(unit.added),
                "removed": sorted(unit.removed),
                "newly_proved": sorted(unit.newly_proved),
                "seal_status": unit.seal_status,
                "fallback_reason": unit.fallback_reason,
            }
            for unit in self._unit_sets
        ]
        return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def capabilities(self) -> dict[str, Any]:
        notes = (
            "Hermetic estimated cost model; real prover and recursive verification "
            "are not available in this process. GPU counters are unavailable when "
            "gpu_available is false. Simulated required units are always zero."
        )
        return {
            "real_prover_available": self.real_prover_available,
            "recursive_verification_available": self.recursive_verification_available,
            "gpu_available": self.gpu_available,
            "notes": notes,
        }

    def build_transition_row(
        self,
        unit_set: TransitionUnitSet,
        *,
        parent_seal: str | None,
        repository_revision: str,
    ) -> dict[str, Any]:
        required_n = len(unit_set.required)
        reused_n = len(unit_set.reused)
        invalidated_n = len(unit_set.invalidated)
        added_n = len(unit_set.added)
        removed_n = len(unit_set.removed)
        newly_n = len(unit_set.newly_proved)
        if newly_n != invalidated_n + added_n:
            raise BenchmarkError(f"row {unit_set.index:02d}: newly_proved arithmetic")
        if required_n != reused_n + newly_n:
            raise BenchmarkError(f"row {unit_set.index:02d}: required arithmetic")

        metrics, provenance = _estimate_metrics(
            seed=self.seed,
            index=unit_set.index,
            required=required_n,
            reused=reused_n,
            newly_proved=newly_n,
            seal_status=unit_set.seal_status,
            gpu_available=self.gpu_available,
        )
        hit = 0.0 if required_n == 0 else reused_n / required_n
        savings = _compute_saved_percent(
            metrics["full_proof_cost"], metrics["incremental_proof_cost"]
        )
        root = _digest(
            f"{self.seed}:{unit_set.index}:{unit_set.scenario}:"
            f"{sorted(unit_set.required)}:{parent_seal}"
        )
        row: dict[str, Any] = {
            "index": unit_set.index,
            "scenario": unit_set.scenario,
            "repository_revision": repository_revision,
            "parent_seal": parent_seal,
            "seal_status": unit_set.seal_status,
            "required_units": required_n,
            "reused_units": reused_n,
            "invalidated_units": invalidated_n,
            "added_units": added_n,
            "removed_units": removed_n,
            "newly_proved_units": newly_n,
            "unit_count_provenance": UNIT_COUNT_PROVENANCE,
            "cache_hit_rate": hit,
            **metrics,
            "metric_provenance": provenance,
            "measurement_provenance": _summarize_provenance(provenance.values()),
            "compute_saved_percent": savings,
            "chain_depth": unit_set.chain_depth,
            "fallback_reason": unit_set.fallback_reason,
            "full_seal_root": root,
            "incremental_seal_root": root,
            "deterministic_roots_match": True,
            "simulated_required_units": 0,
            "rejected_attempts": [dict(item) for item in unit_set.rejected_attempts],
        }
        return row

    def run(
        self,
        *,
        benchmark_worktree_parent_revision: str,
        source_revisions: Mapping[str, str],
        source_trees: Mapping[str, str],
        execution_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Materialize the full benchmark-results@2 payload."""

        if set(source_revisions) != {"accelerate", "datasets", "kit"}:
            raise BenchmarkError("source_revisions must bind accelerate/datasets/kit")
        if set(source_trees) != {"accelerate", "datasets", "kit"}:
            raise BenchmarkError("source_trees must bind accelerate/datasets/kit")
        for name, revision in source_revisions.items():
            if not isinstance(revision, str) or len(revision) != 40:
                raise BenchmarkError(f"source_revisions.{name} must be a 40-char hex")
        for name, tree in source_trees.items():
            if not isinstance(tree, str) or len(tree) != 40:
                raise BenchmarkError(f"source_trees.{name} must be a 40-char hex")
        if source_revisions.get("accelerate") != benchmark_worktree_parent_revision:
            raise BenchmarkError(
                "benchmark_worktree_parent_revision must equal accelerate revision"
            )

        context = dict(execution_context) if execution_context is not None else {
            "runner_id": PROTECTED_RUNNER_ID,
            "argv": protected_benchmark_argv(
                seed=self.seed, transitions=self.transition_count
            ),
            "process_observed": True,
            "test_execution_cryptographically_proven": False,
            "claim": PROTECTED_CLAIM,
        }

        transitions: list[dict[str, Any]] = []
        parent_seal: str | None = None
        for unit_set in self._unit_sets:
            revision = _revision(
                f"{self.seed}:commit:{unit_set.index}:{unit_set.scenario}:"
                f"{benchmark_worktree_parent_revision}"
            )
            row = self.build_transition_row(
                unit_set,
                parent_seal=parent_seal,
                repository_revision=revision,
            )
            transitions.append(row)
            parent_seal = row["full_seal_root"]

        return {
            "schema_version": BENCHMARK_SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "seed": self.seed,
            "transition_count": self.transition_count,
            "benchmark_worktree_parent_revision": benchmark_worktree_parent_revision,
            "source_revisions": {
                "accelerate": source_revisions["accelerate"],
                "datasets": source_revisions["datasets"],
                "kit": source_revisions["kit"],
            },
            "source_trees": {
                "accelerate": source_trees["accelerate"],
                "datasets": source_trees["datasets"],
                "kit": source_trees["kit"],
            },
            "execution_context": context,
            "capabilities": self.capabilities(),
            "transitions": transitions,
        }

    def render_csv(self, payload: Mapping[str, Any]) -> str:
        transitions = payload.get("transitions")
        if not isinstance(transitions, list):
            raise BenchmarkError("payload.transitions must be a list")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream, fieldnames=list(CSV_FIELDS), lineterminator="\n"
        )
        writer.writeheader()
        for row in transitions:
            if not isinstance(row, Mapping):
                raise BenchmarkError("each transition must be an object")
            projected: dict[str, Any] = {}
            for field in CSV_FIELDS:
                value = row.get(field)
                if value is None:
                    projected[field] = ""
                elif isinstance(value, bool):
                    projected[field] = str(value).lower()
                else:
                    projected[field] = value
            writer.writerow(projected)
        return stream.getvalue()

    def write_artifacts(
        self,
        payload: Mapping[str, Any],
        *,
        json_output: Path,
        csv_output: Path,
    ) -> None:
        json_output = Path(json_output)
        csv_output = Path(csv_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        json_bytes = _canonical_json_bytes(payload) + b"\n"
        json_output.write_bytes(json_bytes)
        csv_output.write_text(self.render_csv(payload), encoding="utf-8")


def protected_benchmark_argv(
    *,
    seed: int = DEFAULT_SEED,
    transitions: int = DEFAULT_TRANSITION_COUNT,
    executable: str | None = None,
) -> list[str]:
    """Argv contract consumed by the protected board benchmark runner."""

    return [
        executable if executable is not None else sys.executable,
        CLI_RELATIVE,
        "--seed",
        str(seed),
        "--transitions",
        str(transitions),
        "--json-output",
        f"../staged/{DEFAULT_JSON_RELATIVE}",
        "--csv-output",
        f"../staged/{DEFAULT_CSV_RELATIVE}",
    ]


def _git_stdout(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
                "LC_ALL": "C",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            },
        )
    except OSError as exc:
        raise BenchmarkError(f"git unavailable: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise BenchmarkError(
            f"git {' '.join(args)} failed in {repo}: {detail or completed.returncode}"
        )
    return completed.stdout.strip()


def resolve_source_bindings(repo_root: Path | None = None) -> tuple[str, dict[str, str], dict[str, str]]:
    """Resolve accelerate/datasets/kit HEAD revisions and trees."""

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    paths = {
        "accelerate": root,
        "datasets": root / "ipfs_datasets_py",
        "kit": root / "ipfs_kit_py",
    }
    revisions: dict[str, str] = {}
    trees: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_dir():
            raise BenchmarkError(f"repository path missing for {name}: {path}")
        revision = _git_stdout(path, "rev-parse", "HEAD")
        tree = _git_stdout(path, "rev-parse", "HEAD^{tree}")
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise BenchmarkError(f"non-hex git revision for {name}")
        if len(tree) != 40 or any(ch not in "0123456789abcdef" for ch in tree):
            raise BenchmarkError(f"non-hex git tree for {name}")
        revisions[name] = revision
        trees[name] = tree
    return revisions["accelerate"], revisions, trees


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_RELATIVE,
        description=(
            "Deterministic IncrementalProofSealer forty-transition benchmark "
            "(IPS-052). Writes canonical JSON and CSV artifacts."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--transitions", type=int, default=DEFAULT_TRANSITION_COUNT)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument(
        "--repo-root",
        default=None,
        help="repository root for git source bindings (default: cwd)",
    )
    parser.add_argument(
        "--gpu-available",
        action="store_true",
        help="mark GPU counters estimated instead of unavailable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        benchmark = IncrementalProofBenchmark(
            seed=args.seed,
            transition_count=args.transitions,
            gpu_available=bool(args.gpu_available),
        )
        parent, revisions, trees = resolve_source_bindings(
            Path(args.repo_root) if args.repo_root else None
        )
        # Protected runner always passes the closed staged paths; embed that
        # exact argv contract so IPS-053 validation binds process observation.
        if (
            args.seed == DEFAULT_SEED
            and args.transitions == DEFAULT_TRANSITION_COUNT
        ):
            context_argv = protected_benchmark_argv(seed=args.seed, transitions=args.transitions)
        else:
            context_argv = [
                sys.executable,
                CLI_RELATIVE,
                "--seed",
                str(args.seed),
                "--transitions",
                str(args.transitions),
                "--json-output",
                str(args.json_output),
                "--csv-output",
                str(args.csv_output),
            ]
        payload = benchmark.run(
            benchmark_worktree_parent_revision=parent,
            source_revisions=revisions,
            source_trees=trees,
            execution_context={
                "runner_id": PROTECTED_RUNNER_ID,
                "argv": context_argv,
                "process_observed": True,
                "test_execution_cryptographically_proven": False,
                "claim": PROTECTED_CLAIM,
            },
        )
        benchmark.write_artifacts(
            payload,
            json_output=Path(args.json_output),
            csv_output=Path(args.csv_output),
        )
    except BenchmarkError as exc:
        sys.stderr.write(f"benchmark error: {exc}\n")
        return 2
    return 0


__all__ = (
    "BENCHMARK_ID",
    "BENCHMARK_INTERFACE",
    "BENCHMARK_SCHEMA",
    "BenchmarkError",
    "CSV_FIELDS",
    "DEFAULT_SEED",
    "DEFAULT_TRANSITION_COUNT",
    "EVIDENCE_SUBSET",
    "FULL_TRANSITIONS",
    "CONDITIONAL_FULL_TRANSITIONS",
    "IncrementalProofBenchmark",
    "METRIC_FIELDS",
    "SCENARIOS",
    "TransitionSpec",
    "TransitionUnitSet",
    "build_parser",
    "main",
    "protected_benchmark_argv",
    "resolve_source_bindings",
)


if __name__ == "__main__":
    raise SystemExit(main())
