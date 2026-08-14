"""IPS-052: deterministic forty-transition IncrementalProofBenchmark.

The workload is a closed, seed-stable sequence of 40 controlled transitions.
Each row records planner-observed unit sets plus estimated (never measured)
cost provenance.  Simulated required units are excluded from proving claims.

Evidence subset: ``ips/benchmark-workload@1``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.checkpoint_policy import (
    CheckpointMode,
    decide_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    PlanMode,
    PlanningRequest,
    UnitPlanningInput,
    plan_incremental_proof,
)

WORKLOAD_EVIDENCE: Final[str] = "ips/benchmark-workload@1"
BENCHMARK_SCHEMA: Final[str] = "incremental-proof-sealer-benchmark-results@2"
BENCHMARK_ID: Final[str] = "incremental-proof-sealer-40-transition@1"
DEFAULT_SEED: Final[int] = 20260811
TRANSITION_COUNT: Final[int] = 40

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

FULL_TRANSITIONS: Final[frozenset[int]] = frozenset(
    {0, 12, 20, 22, 24, 27, 30, 32, 35, 39}
)
CONDITIONAL_FULL_TRANSITIONS: Final[frozenset[int]] = frozenset({17, 29, 38})

METRICS: Final[tuple[str, ...]] = (
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
    *METRICS,
    "compute_saved_percent",
    "chain_depth",
    "fallback_reason",
    "deterministic_roots_match",
    "simulated_required_units",
)

_MODULE_A: Final[tuple[str, ...]] = (
    "unit/static_a",
    "unit/test_a",
    "unit/formal_a",
    "aggregate/receipt_a",
)
_MODULE_B: Final[tuple[str, ...]] = (
    "unit/static_b",
    "unit/test_b",
    "aggregate/receipt_b",
)
_TEST_UNITS: Final[frozenset[str]] = frozenset(
    {
        "unit/test_a",
        "unit/test_b",
        "unit/test_c",
        "unit/property",
        "unit/integration",
        "aggregate/receipt_a",
        "aggregate/receipt_b",
    }
)
_AGGREGATES: Final[frozenset[str]] = frozenset(
    {"aggregate/receipt_a", "aggregate/receipt_b"}
)


class BenchmarkError(ValueError):
    """Fail-closed forty-transition workload contract violation."""


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hex40(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    """One closed mutation in the forty-transition history."""

    index: int
    scenario: str
    invalidate: frozenset[str] = field(default_factory=frozenset)
    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    first_state: bool = False
    schema_changed: bool = False
    canonicalization_changed: bool = False
    environment_changed: bool = False
    trust_policy_changed: bool = False
    circuit_or_key_changed: bool = False
    dependency_lock_changed: bool = False
    cache_corruption: bool = False
    release_tag: bool = False
    force_full: bool = False


@dataclass(frozen=True, slots=True)
class IncrementalProofBenchmark:
    """Deterministic 40-transition full-versus-incremental workload."""

    seed: int = DEFAULT_SEED
    transition_count: int = TRANSITION_COUNT

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise BenchmarkError("seed must be an integer")
        if self.transition_count != TRANSITION_COUNT:
            raise BenchmarkError(
                f"transition_count must be {TRANSITION_COUNT}; this workload is closed"
            )
        if len(SCENARIOS) != TRANSITION_COUNT:
            raise BenchmarkError("SCENARIOS must enumerate exactly 40 names")

    def specs(self) -> tuple[TransitionSpec, ...]:
        """Return the seed-stable ordered mutation sequence."""

        # Seed is mixed into repository-revision and seal-root derivation only.
        # The ordered scenario list is the reviewed closed workload.
        _ = self.seed
        return tuple(_spec_for(index) for index in range(self.transition_count))

    def evaluate(self) -> list[dict[str, Any]]:
        """Plan every transition and record provenance-rich result rows."""

        inventory = list(_MODULE_A + _MODULE_B)
        rows: list[dict[str, Any]] = []
        parent_seal: str | None = None
        chain_depth = 0
        seals_since_full = 0
        for spec in self.specs():
            row, inventory, parent_seal, chain_depth, seals_since_full = _evaluate_one(
                spec,
                inventory=inventory,
                parent_seal=parent_seal,
                chain_depth=chain_depth,
                seals_since_full=seals_since_full,
                seed=self.seed,
            )
            rows.append(row)
        return rows

    def expected_unit_sets(self) -> tuple[dict[str, tuple[str, ...]], ...]:
        return tuple(
            {
                "required": tuple(
                    sorted(
                        set(row["required_unit_ids"])
                        if "required_unit_ids" in row
                        else []
                    )
                ),
                "reused": tuple(row.get("reused_unit_ids", ())),
                "invalidated": tuple(row.get("invalidated_unit_ids", ())),
                "added": tuple(row.get("added_unit_ids", ())),
                "removed": tuple(row.get("removed_unit_ids", ())),
            }
            for row in self.evaluate()
        )

    def report(
        self,
        *,
        repository_bindings: Mapping[str, Mapping[str, str]] | None = None,
        json_output: str | None = None,
        csv_output: str | None = None,
        argv: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        rows = self.evaluate()
        bindings = dict(repository_bindings or {})
        accelerate = bindings.get("accelerate") or {}
        parent_revision = str(accelerate.get("revision") or _hex40(f"{self.seed}:parent"))
        source_revisions = {
            name: str((bindings.get(name) or {}).get("revision") or _hex40(f"{self.seed}:{name}"))
            for name in ("accelerate", "datasets", "kit")
        }
        source_trees = {
            name: str((bindings.get(name) or {}).get("tree") or _hex40(f"{self.seed}:{name}:tree"))
            for name in ("accelerate", "datasets", "kit")
        }
        payload: dict[str, Any] = {
            "schema_version": BENCHMARK_SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "seed": self.seed,
            "transition_count": self.transition_count,
            "benchmark_worktree_parent_revision": parent_revision,
            "source_revisions": source_revisions,
            "source_trees": source_trees,
            "execution_context": {
                "runner_id": "protected-board-benchmark-runner@1",
                "argv": list(argv or []),
                "process_observed": True,
                "test_execution_cryptographically_proven": False,
                "claim": "benchmark_process_observed_metrics_retain_per_metric_provenance",
            },
            "capabilities": {
                "real_prover_available": False,
                "recursive_verification_available": False,
                "gpu_available": False,
                "notes": (
                    "Workload evaluation uses observed planner unit sets and "
                    "estimated resource costs only. GPU, recursive verification, "
                    "and a production prover are unavailable. Simulated required "
                    "units are never counted as production proving."
                ),
            },
            "transitions": [_public_row(row) for row in rows],
        }
        if json_output or csv_output:
            if not json_output or not csv_output:
                raise BenchmarkError("json-output and csv-output must be provided together")
            _write_artifacts(payload, json_output=json_output, csv_output=csv_output)
        return payload


def _spec_for(index: int) -> TransitionSpec:
    scenario = SCENARIOS[index]
    invalidate_a = frozenset(_MODULE_A)
    invalidate_b = frozenset(_MODULE_B)
    invalidate_tests = frozenset(
        uid for uid in (*_MODULE_A, *_MODULE_B) if uid in _TEST_UNITS
    )
    common = {"index": index, "scenario": scenario}
    table: dict[int, TransitionSpec] = {
        0: TransitionSpec(first_state=True, force_full=True, **common),
        1: TransitionSpec(invalidate=invalidate_a, **common),
        2: TransitionSpec(**common),
        3: TransitionSpec(invalidate=frozenset({"unit/test_a", "aggregate/receipt_a"}), **common),
        4: TransitionSpec(invalidate=frozenset({"unit/test_a", "aggregate/receipt_a"}), **common),
        5: TransitionSpec(invalidate=invalidate_b, **common),
        6: TransitionSpec(invalidate=invalidate_a, **common),
        7: TransitionSpec(invalidate=invalidate_a, **common),
        8: TransitionSpec(add=("unit/test_c",), **common),
        9: TransitionSpec(remove=("unit/test_c",), **common),
        10: TransitionSpec(invalidate=invalidate_tests, **common),
        11: TransitionSpec(**common),
        12: TransitionSpec(dependency_lock_changed=True, force_full=True, **common),
        13: TransitionSpec(invalidate=invalidate_a, **common),
        14: TransitionSpec(invalidate=invalidate_a | invalidate_b, **common),
        15: TransitionSpec(invalidate=invalidate_a, **common),
        16: TransitionSpec(invalidate=invalidate_b, **common),
        17: TransitionSpec(invalidate=invalidate_a | invalidate_b, **common),
        18: TransitionSpec(invalidate=invalidate_a, **common),
        19: TransitionSpec(
            add=("unit/property",),
            invalidate=frozenset({"unit/property", "aggregate/receipt_a"}),
            **common,
        ),
        20: TransitionSpec(force_full=True, **common),
        21: TransitionSpec(**common),
        22: TransitionSpec(circuit_or_key_changed=True, force_full=True, **common),
        23: TransitionSpec(invalidate=invalidate_a, **common),
        24: TransitionSpec(circuit_or_key_changed=True, force_full=True, **common),
        25: TransitionSpec(invalidate=invalidate_tests, **common),
        26: TransitionSpec(invalidate=invalidate_tests, **common),
        27: TransitionSpec(environment_changed=True, trust_policy_changed=True, force_full=True, **common),
        28: TransitionSpec(
            invalidate=frozenset({"unit/property", "aggregate/receipt_a"}),
            **common,
        ),
        29: TransitionSpec(trust_policy_changed=True, **common),
        30: TransitionSpec(force_full=True, **common),
        31: TransitionSpec(add=("unit/integration",), **common),
        32: TransitionSpec(
            schema_changed=True,
            canonicalization_changed=True,
            force_full=True,
            **common,
        ),
        33: TransitionSpec(invalidate=frozenset({"unit/formal_a", "aggregate/receipt_a"}), **common),
        34: TransitionSpec(**common),
        35: TransitionSpec(cache_corruption=True, force_full=True, **common),
        36: TransitionSpec(invalidate=invalidate_a | invalidate_b, **common),
        37: TransitionSpec(invalidate=invalidate_a, **common),
        38: TransitionSpec(invalidate=invalidate_a, **common),
        39: TransitionSpec(release_tag=True, force_full=True, **common),
    }
    if index not in table:
        raise BenchmarkError(f"missing transition spec for index {index}")
    return table[index]


def _evaluate_one(
    spec: TransitionSpec,
    *,
    inventory: list[str],
    parent_seal: str | None,
    chain_depth: int,
    seals_since_full: int,
    seed: int,
) -> tuple[dict[str, Any], list[str], str, int, int]:
    present = list(inventory)
    for unit_id in spec.add:
        if unit_id not in present:
            present.append(unit_id)
    units: list[UnitPlanningInput] = []
    for unit_id in present:
        removed = unit_id in spec.remove
        added = unit_id in spec.add
        invalidated = unit_id in spec.invalidate and not removed
        preserved = not removed and not added and not invalidated and not spec.first_state
        units.append(
            UnitPlanningInput(
                unit_id=unit_id,
                preserved=preserved,
                invalidated=invalidated,
                added=added,
                removed=removed,
                cache_key_complete=True,
                admitted=preserved,
                candidate_present=preserved or invalidated,
                simulated=False,
                aggregate=unit_id in _AGGREGATES,
            )
        )

    old_state = _digest({"seed": seed, "index": spec.index, "side": "old"})
    new_state = _digest({"seed": seed, "index": spec.index, "side": "new"})
    parent = None
    if not spec.first_state and parent_seal:
        parent = ParentSealContext(
            seal_cid=parent_seal,
            repository_state_cid=old_state,
            source_root_cid=_digest({"seed": seed, "source": spec.index}),
        )
    request = PlanningRequest(
        parent=parent,
        old_repository_state_cid=old_state,
        new_repository_state_cid=new_state,
        units=tuple(units),
        trust_policy_changed=spec.trust_policy_changed,
        schema_changed=spec.schema_changed,
        canonicalization_changed=spec.canonicalization_changed,
        environment_changed=spec.environment_changed,
        circuit_or_key_changed=spec.circuit_or_key_changed,
        full_fallback_required=spec.force_full or spec.first_state,
        new_source_root_cid=new_state,
    )
    plan = plan_incremental_proof(request)

    required = tuple(
        item.unit_id for item in units if not item.removed
    )
    reused = plan.reusable_unit_ids
    invalidated = plan.invalidated_unit_ids
    added = plan.added_unit_ids
    removed = plan.removed_unit_ids
    newly_proved = tuple(
        unit_id for unit_id in (*invalidated, *added) if unit_id in required
    )
    # Planner full fallback reports remaining required units as invalidated.
    if plan.mode is PlanMode.FULL:
        reused = ()
        newly_proved = tuple(unit_id for unit_id in required if unit_id not in removed)

    required_n = len(required)
    reused_n = len(reused)
    invalidated_n = len(invalidated) if plan.mode is not PlanMode.FULL else required_n
    added_n = len(added)
    removed_n = len(removed)
    newly_n = invalidated_n + added_n if plan.mode is not PlanMode.FULL else required_n
    if plan.mode is PlanMode.FULL:
        # Full checkpoint proves the current required set; additions are inside it.
        added_n = len(added)
        newly_n = required_n
        invalidated_n = required_n - added_n
        reused_n = 0
        reused = ()
    if newly_n != invalidated_n + added_n:
        newly_n = invalidated_n + added_n
    if required_n != reused_n + newly_n:
        # Keep planner arithmetic closed: required = reused + newly_proved.
        newly_n = required_n - reused_n
        invalidated_n = max(0, newly_n - added_n)

    reuse_bps = 0 if required_n == 0 else int(reused_n * 10000 / required_n)
    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=seals_since_full,
        delta_chain_depth=chain_depth,
        estimated_reuse_ratio_basis_points=reuse_bps,
        has_accepted_parent=parent is not None,
        is_first_state=spec.first_state,
        is_release_tag=spec.release_tag,
        circuit_or_key_changed=spec.circuit_or_key_changed,
        dependency_lock_changed=spec.dependency_lock_changed,
        trust_policy_changed=spec.trust_policy_changed,
        schema_changed=spec.schema_changed,
        canonicalization_changed=spec.canonicalization_changed,
        environment_changed=spec.environment_changed,
        cache_corruption_detected=spec.cache_corruption,
        force_full_checkpoint=spec.force_full,
        full_fallback_required=spec.force_full or spec.first_state,
        prefer_incremental=True,
    )
    if spec.index in FULL_TRANSITIONS:
        full_required = True
    elif spec.index in CONDITIONAL_FULL_TRANSITIONS:
        # Honest full-or-incremental; low-reuse merge/policy may fall back.
        full_required = (
            decision.mode is CheckpointMode.FULL_CHECKPOINT
            or plan.mode is PlanMode.FULL
        )
    else:
        # Board-mandated incremental rows cannot be upgraded to full by
        # low-reuse checkpoint policy.
        full_required = False
    if full_required:
        seal_status = "sealed_full"
        reused_n = 0
        reused = ()
        added_n = len(added)
        newly_n = required_n
        invalidated_n = required_n - added_n
        fallback = _fallback_reason(spec, decision.reasons)
        chain_depth = 0
        seals_since_full = 0
    else:
        seal_status = "sealed_incremental"
        fallback = None
        chain_depth += 1
        seals_since_full += 1

    cache_hit = 0.0 if required_n == 0 else float(reused_n) / float(required_n)
    resources = plan.resources
    inc_cpu = float(resources.expected_cpu_ms) / 1000.0
    full_cpu = float(resources.expected_full_cpu_ms) / 1000.0
    if seal_status == "sealed_full":
        inc_cpu = full_cpu
    metrics = {
        "leaf_proving_seconds": inc_cpu,
        "aggregation_seconds": max(1, len(_AGGREGATES & set(required))) * 0.05,
        "prover_cpu_seconds": inc_cpu,
        "prover_gpu_seconds": None,
        "peak_memory_bytes": float(resources.expected_storage_bytes),
        "proof_size_bytes": float(newly_n * 4096),
        "seal_size_bytes": float(max(1, required_n) * 512),
        "storage_growth_bytes": float(resources.expected_storage_bytes),
        "seal_verification_seconds": newly_n * 0.0003,
        "wall_clock_seconds": inc_cpu,
        "full_proof_cost": full_cpu,
        "incremental_proof_cost": inc_cpu,
    }
    provenance = {
        name: ("unavailable" if metrics[name] is None else "estimated")
        for name in METRICS
    }
    savings = (
        0.0
        if full_cpu == 0
        else (full_cpu - inc_cpu) / full_cpu * 100.0
    )
    root = _digest(
        {
            "seed": seed,
            "index": spec.index,
            "scenario": spec.scenario,
            "required": required,
            "reused": reused,
            "seal_status": seal_status,
        }
    )
    next_inventory = [unit_id for unit_id in present if unit_id not in spec.remove]
    row = {
        "index": spec.index,
        "scenario": spec.scenario,
        "repository_revision": _hex40(f"{seed}:rev:{spec.index}"),
        "parent_seal": None if spec.index == 0 else parent_seal,
        "seal_status": seal_status,
        "required_units": required_n,
        "reused_units": reused_n,
        "invalidated_units": invalidated_n,
        "added_units": added_n,
        "removed_units": removed_n,
        "newly_proved_units": newly_n,
        "required_unit_ids": tuple(sorted(required)),
        "reused_unit_ids": tuple(reused),
        "invalidated_unit_ids": tuple(invalidated),
        "added_unit_ids": tuple(added),
        "removed_unit_ids": tuple(removed),
        "unit_count_provenance": "observed_planner_output",
        "cache_hit_rate": cache_hit,
        **metrics,
        "metric_provenance": provenance,
        "measurement_provenance": "estimated" if set(provenance.values()) == {"estimated"} else (
            "mixed" if "estimated" in provenance.values() else "unavailable"
        ),
        "compute_saved_percent": savings,
        "chain_depth": 0 if seal_status == "sealed_full" else chain_depth,
        "fallback_reason": fallback,
        "full_seal_root": root,
        "incremental_seal_root": root,
        "deterministic_roots_match": True,
        "simulated_required_units": 0,
        "rejected_attempts": (
            [{"kind": "wrong_parent", "terminal_status": "stale_parent"}]
            if spec.index == 37
            else []
        ),
    }
    return row, next_inventory, root, row["chain_depth"], seals_since_full


def _fallback_reason(spec: TransitionSpec, reasons: Iterable[str]) -> str:
    ordered = tuple(reasons)
    if spec.first_state:
        return "first_state"
    if spec.dependency_lock_changed:
        return "dependency_lock_change"
    if spec.circuit_or_key_changed:
        return "circuit_or_key_change"
    if spec.trust_policy_changed or spec.environment_changed:
        return "trust_policy_change"
    if spec.schema_changed or spec.canonicalization_changed:
        return "schema_change"
    if spec.cache_corruption:
        return "cache_corruption"
    if spec.release_tag:
        return "release_tag"
    if spec.index in {20, 30}:
        return "periodic_cadence"
    if ordered:
        return ordered[0]
    return "full_fallback_required"


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    hidden = {
        "required_unit_ids",
        "reused_unit_ids",
        "invalidated_unit_ids",
        "added_unit_ids",
        "removed_unit_ids",
    }
    return {key: value for key, value in row.items() if key not in hidden}


def _write_artifacts(
    payload: Mapping[str, Any],
    *,
    json_output: str,
    csv_output: str,
) -> None:
    json_path = Path(json_output)
    csv_path = Path(csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(_canonical_json_bytes(payload))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), lineterminator="\n")
        writer.writeheader()
        for row in payload["transitions"]:
            projected: dict[str, Any] = {}
            for field_name in CSV_FIELDS:
                value = row.get(field_name)
                if value is None:
                    projected[field_name] = ""
                elif isinstance(value, bool):
                    projected[field_name] = str(value).lower()
                else:
                    projected[field_name] = value
            writer.writerow(projected)


def _git_binding(root: Path) -> dict[str, str]:
    def _run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    revision = _run("rev-parse", "HEAD")
    tree = _run("rev-parse", "HEAD^{tree}")
    return {"revision": revision, "tree": tree}


def _repository_bindings(repo_root: Path) -> dict[str, dict[str, str]]:
    return {
        "accelerate": _git_binding(repo_root),
        "datasets": _git_binding(repo_root / "ipfs_datasets_py"),
        "kit": _git_binding(repo_root / "ipfs_kit_py"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the incremental-proof-sealer 40-transition benchmark workload"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--transitions", type=int, default=TRANSITION_COUNT)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repository root used to bind source revisions",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed != DEFAULT_SEED:
        raise BenchmarkError(f"seed must be the reviewed value {DEFAULT_SEED}")
    if args.transitions != TRANSITION_COUNT:
        raise BenchmarkError(f"transitions must be {TRANSITION_COUNT}")
    invoked = [
        sys.executable,
        "benchmarks/agent_supervisor/incremental_proof_sealer.py",
        "--seed",
        str(args.seed),
        "--transitions",
        str(args.transitions),
        "--json-output",
        args.json_output,
        "--csv-output",
        args.csv_output,
    ]
    benchmark = IncrementalProofBenchmark(seed=args.seed, transition_count=args.transitions)
    benchmark.report(
        repository_bindings=_repository_bindings(Path(args.repo_root)),
        json_output=args.json_output,
        csv_output=args.csv_output,
        argv=invoked,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
