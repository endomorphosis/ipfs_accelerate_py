"""Qualification writer for ASEH-070..073 without recaiming ASEH-013 outputs.

The sealed ASEH-013 harness only understands ``--write``/``--check``. Later
qualification tasks declare a ``--cohort``/``--output`` CLI against that same
path. Implementers then rewrite ``paired_harness.py`` (013's declared output)
and the supervisor stalls on extra-path restore.

This module is the deterministic writer those board commands need. The 013
``main()`` forwards unknown qualification flags here; 013 ``--write``/``--check``
and default campaign printing stay unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    EQUAL_CONTROL_FIELDS,
    HERMETIC_MINIMUM,
    HISTORICAL_MINIMUM,
    LIVE_MINIMUM,
    PAIRED_ARMS,
    PROGRAM_ID,
    STATISTIC_FIELDS,
    collect_unavailable_fields,
    content_identity,
    measured_quantity,
    unavailable,
)


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
HARNESS_PATH: Final[Path] = (
    REPO_ROOT
    / "benchmarks"
    / "agent_supervisor"
    / "efficiency_state_hardening"
    / "paired_harness.py"
)
QUALIFICATION_OBJECTIVE_ID: Final[str] = "ASEH-G080"
POLICY_IDENTITY: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
)
CLOSED_DISPOSITIONS: Final[tuple[str, ...]] = (
    "evidence_qualified",
    "insufficient_evidence",
    "safety_or_quality_failed",
    "not_admitted",
)
COHORT_TASK: Final[dict[str, str]] = {
    "hermetic": "ASEH-070",
    "historical": "ASEH-071",
    "live-shadow": "ASEH-072",
    "canary": "ASEH-073",
}
COHORT_MINIMUM: Final[dict[str, int]] = {
    "hermetic": HERMETIC_MINIMUM,
    "historical": HISTORICAL_MINIMUM,
    "live-shadow": LIVE_MINIMUM,
    "canary": 1,
}
COHORT_POPULATION: Final[dict[str, str]] = {
    "hermetic": "hermetic_development",
    "historical": "historical_exact_tree_replay",
    "live-shadow": "new_live_shadow_canary",
    "canary": "new_live_shadow_canary",
}
CRITICAL_SEED_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"failed", "conflicted", "human_escalated"}
)
BEHAVIOR_COVERAGE: Final[dict[str, tuple[str, ...]]] = {
    "rapid_iteration": ("schema_change", "bug_repair", "feature_addition"),
    "selected_test_false_negatives": ("test_selection",),
    "safety_seeds": ("human_escalation", "merge_conflict", "recovery_replay"),
    "routing": ("routing_policy",),
    "context_pack": ("context_pack_build",),
    "state": ("state_migration", "recovery_replay"),
    "planning": ("proof_obligation",),
    "synthesis": ("feature_addition", "schema_change"),
}


class QualificationCliError(RuntimeError):
    """Fail-closed qualification writer error."""


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aseh_013_paired_harness_qualification",
        HARNESS_PATH,
    )
    if spec is None or spec.loader is None:
        raise QualificationCliError("ASEH-013 paired harness is unreadable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_identity(repo_root: Path) -> tuple[str, str]:
    def _run(args: Sequence[str]) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise QualificationCliError(
                f"git {' '.join(args)} failed: {(completed.stderr or completed.stdout).strip()}"
            )
        return completed.stdout.strip()

    return _run(["rev-parse", "HEAD"]), _run(["rev-parse", "HEAD^{tree}"])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(target)


def _behavior_coverage(distribution: Mapping[str, Any]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for behavior, classes in BEHAVIOR_COVERAGE.items():
        present = {name: int(distribution.get(name, 0) or 0) for name in classes}
        coverage[behavior] = {
            "complete": all(value > 0 for value in present.values()),
            "present_counts": present,
            "required_classes": list(classes),
        }
    return coverage


def _evaluate_hermetic_quality_safety(campaign: Mapping[str, Any]) -> dict[str, Any]:
    recipes = {
        str(item["fixture_id"]): item for item in campaign.get("recipes") or ()
    }
    escaped_seeds: list[dict[str, Any]] = []
    observed_selected_fns: list[str] = []
    escaped_selected_fns: list[str] = []
    simulated_as_live: list[str] = []
    for item in campaign.get("paired_fixtures") or ():
        fixture_id = str(item.get("fixture_id") or "")
        recipe = recipes.get(fixture_id) or {}
        if item.get("live") is True:
            simulated_as_live.append(fixture_id)
        observations = item.get("observations") or {}
        accepted_any = False
        for arm_id, observation in observations.items():
            if not isinstance(observation, Mapping):
                continue
            if observation.get("live") is True:
                simulated_as_live.append(f"{fixture_id}:{arm_id}")
            accepted = observation.get("accepted") is True
            accepted_any = accepted_any or accepted
            if recipe.get("outcome") in CRITICAL_SEED_OUTCOMES and accepted:
                escaped_seeds.append(
                    {
                        "accepted": True,
                        "arm_id": arm_id,
                        "critical": True,
                        "expected_outcome": recipe.get("outcome"),
                        "fixture_id": fixture_id,
                        "seed": int(recipe.get("seed") or 0),
                    }
                )
        if recipe.get("task_class") == "test_selection" and recipe.get("outcome") == "failed":
            observed_selected_fns.append(fixture_id)
            if accepted_any:
                escaped_selected_fns.append(fixture_id)
    return {
        "escaped_seeds": escaped_seeds,
        "escaped_selected_test_false_negatives": escaped_selected_fns,
        "observed_selected_test_false_negatives": observed_selected_fns,
        "simulated_as_live": simulated_as_live,
    }


def _disposition(
    *,
    pair_count: int,
    minimum_tasks: int,
    coverage: Mapping[str, Any],
    quality_safety: Mapping[str, Any],
    not_admitted_reason: str = "",
) -> str:
    if not_admitted_reason:
        return "not_admitted"
    if (
        quality_safety.get("escaped_seeds")
        or quality_safety.get("escaped_selected_test_false_negatives")
        or quality_safety.get("simulated_as_live")
    ):
        return "safety_or_quality_failed"
    behaviors_complete = all(
        item.get("complete") is True for item in coverage.values()
    ) if coverage else False
    if pair_count < minimum_tasks or not behaviors_complete:
        return "insufficient_evidence"
    return "evidence_qualified"


def _bind_identity(payload: dict[str, Any]) -> dict[str, Any]:
    payload["explicit_unavailable_fields"] = list(collect_unavailable_fields(payload))
    body = {key: value for key, value in payload.items() if key != "identity"}
    payload["identity"] = content_identity(body)
    return payload


def _base_receipt(
    *,
    cohort: str,
    disposition: str,
    pair_count: int,
    minimum_tasks: int,
    commit: str,
    tree: str,
    validator_command: Sequence[str],
    allow_honest_nonpromotion: bool,
    allow_not_admitted: bool,
    reasons: Sequence[str],
    quality_safety: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
    statistics: Mapping[str, Any] | None = None,
    audit_overhead: Mapping[str, Any] | None = None,
    results_identity: str = "",
    live: bool = False,
) -> dict[str, Any]:
    task_id = COHORT_TASK[cohort]
    sensor_id = f"aseh-{task_id[-3:]}-{cohort}-qualification"
    qualified = disposition == "evidence_qualified"
    safety = quality_safety or {
        "escaped_seeds": [],
        "escaped_selected_test_false_negatives": [],
        "observed_selected_test_false_negatives": [],
        "simulated_as_live": [],
    }
    payload: dict[str, Any] = {
        "admitted": True,
        "allow_honest_nonpromotion": allow_honest_nonpromotion is True,
        "allow_not_admitted": allow_not_admitted is True,
        "arms": list(PAIRED_ARMS),
        "audit_overhead": dict(audit_overhead or unavailable("not_yet_measured")),
        "authority": False,
        "behavior_coverage": dict(coverage or {}),
        "canary_admitted": False,
        "cohort": cohort,
        "controls": dict(controls or {}),
        "disposition": disposition,
        "equal_control_fields": list(EQUAL_CONTROL_FIELDS),
        "escaped_seeds": list(safety.get("escaped_seeds") or ()),
        "hermetic_sufficient_for_production_promotion": False,
        "interface": "AsehPairedQualification@1",
        "live": live is True,
        "minimum_tasks": int(minimum_tasks),
        "objective_id": QUALIFICATION_OBJECTIVE_ID,
        "policy_identity": POLICY_IDENTITY,
        "population_kind": COHORT_POPULATION[cohort],
        "production_qualified": False,
        "program_id": PROGRAM_ID,
        "promotion_authorized": False,
        "qualification": qualified,
        "quality": {
            "escaped_selected_test_false_negatives": list(
                safety.get("escaped_selected_test_false_negatives") or ()
            ),
            "observed_selected_test_false_negatives": list(
                safety.get("observed_selected_test_false_negatives") or ()
            ),
        },
        "reasons": list(reasons),
        "repository_commit": commit,
        "repository_tree": tree,
        "results_identity": results_identity or "unavailable",
        "safety": {
            "escaped_critical_seeded_defects": measured_quantity(
                len(safety.get("escaped_seeds") or ()),
                unit="count",
                sensor_id=sensor_id,
            )
            if pair_count
            else unavailable("not_yet_measured"),
            "live_cohort": unavailable("fixture_only")
            if cohort == "hermetic"
            else unavailable("not_yet_measured"),
            "simulated_as_live_outcomes": measured_quantity(
                len(safety.get("simulated_as_live") or ()),
                unit="count",
                sensor_id=sensor_id,
            )
            if pair_count
            else unavailable("not_yet_measured"),
        },
        "schema": f"ipfs_accelerate_py/agent-supervisor/aseh-{cohort}-qualification@1",
        "schema_version": 1,
        "statistics": dict(statistics or {}),
        "task_id": task_id,
        "validator_command": [str(item) for item in validator_command],
    }
    payload["pair_count"] = (
        measured_quantity(pair_count, unit="count", sensor_id=sensor_id)
        if pair_count
        else unavailable("not_yet_measured")
    )
    return _bind_identity(payload)


def _results_payload(
    *,
    cohort: str,
    disposition: str,
    pair_count: int,
    minimum_tasks: int,
    allow_honest_nonpromotion: bool,
    allow_not_admitted: bool,
    live: bool,
    controls: Mapping[str, Any] | None = None,
    statistics: Mapping[str, Any] | None = None,
    audit_overhead: Mapping[str, Any] | None = None,
    reasons: Sequence[str] = (),
    vectors_identity: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "allow_honest_nonpromotion": allow_honest_nonpromotion is True,
        "allow_not_admitted": allow_not_admitted is True,
        "arms": list(PAIRED_ARMS),
        "audit_overhead": dict(audit_overhead or unavailable("not_yet_measured")),
        "authority": False,
        "cohort": cohort,
        "controls": dict(controls or {}),
        "disposition": disposition,
        "equal_control_fields": list(EQUAL_CONTROL_FIELDS),
        "fixture_count": int(pair_count),
        "hermetic_sufficient_for_production_promotion": False,
        "interface": "AsehPairedCampaignResults@1",
        "live": live is True,
        "minimum_tasks": int(minimum_tasks),
        "pair_count": int(pair_count),
        "population_kind": COHORT_POPULATION[cohort],
        "promotion_without_paired_campaign": False,
        "reasons": list(reasons),
        "required_statistics": list(STATISTIC_FIELDS),
        "schema": f"ipfs_accelerate_py/agent-supervisor/aseh-{cohort}-campaign-results@1",
        "schema_version": 1,
        "statistics": dict(statistics or {}),
        "task_id": COHORT_TASK[cohort],
        "vectors_identity": vectors_identity or "unavailable",
    }
    return _bind_identity(payload)


def _run_hermetic(
    harness: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]:
    recipes = harness.load_vectors()
    campaign = harness.run_paired_campaign(recipes)
    if campaign.get("live") is True:
        raise QualificationCliError("hermetic evidence cannot be represented as live")
    quality_safety = _evaluate_hermetic_quality_safety(campaign)
    stats = campaign.get("statistics") or {}
    coverage = _behavior_coverage(stats.get("distribution_by_task_class") or {})
    pair_count = int(campaign.get("fixture_count") or 0)
    return campaign, quality_safety, pair_count, coverage


def run_qualification(
    *,
    cohort: str,
    output: Path,
    qualification_output: Path,
    minimum_tasks: int | None = None,
    allow_honest_nonpromotion: bool = False,
    allow_not_admitted: bool = False,
    require_shadow_receipt: Path | None = None,
    validator_command: Sequence[str] = (),
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Write campaign results and a qualification receipt for one cohort."""

    if cohort not in COHORT_TASK:
        raise QualificationCliError(f"unsupported cohort {cohort!r}")
    bound_minimum = int(minimum_tasks or COHORT_MINIMUM[cohort])
    root = Path(repo_root or REPO_ROOT)
    commit, tree = _git_identity(root)
    command = [str(item) for item in validator_command]
    not_admitted_reason = ""
    campaign: Mapping[str, Any] = {}
    quality_safety: dict[str, Any] = {
        "escaped_seeds": [],
        "escaped_selected_test_false_negatives": [],
        "observed_selected_test_false_negatives": [],
        "simulated_as_live": [],
    }
    coverage: dict[str, Any] = {}
    pair_count = 0
    live = False
    controls: Mapping[str, Any] = {}
    statistics: Mapping[str, Any] = {}
    audit_overhead: Mapping[str, Any] = {}
    vectors_identity = ""
    reasons: list[str] = []

    if cohort == "canary":
        if require_shadow_receipt is None:
            not_admitted_reason = "missing_shadow_receipt"
        else:
            shadow_path = Path(require_shadow_receipt)
            if not shadow_path.is_file():
                not_admitted_reason = "missing_shadow_receipt"
            else:
                try:
                    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    not_admitted_reason = "unreadable_shadow_receipt"
                    shadow = {}
                if not isinstance(shadow, Mapping):
                    not_admitted_reason = "unreadable_shadow_receipt"
                elif shadow.get("disposition") != "evidence_qualified":
                    not_admitted_reason = "shadow_not_qualified"
        if not_admitted_reason:
            reasons = [not_admitted_reason, "zero_candidate_mutation"]
    elif cohort == "hermetic":
        if require_shadow_receipt is not None:
            raise QualificationCliError("hermetic qualification cannot consume a shadow receipt")
        harness = _load_harness()
        campaign, quality_safety, pair_count, coverage = _run_hermetic(harness)
        controls = campaign.get("controls") or {}
        statistics = {
            key: campaign["statistics"][key]
            for key in (
                "median_difference",
                "mean_difference",
                "accepted_patch_rate",
                "quality_adjusted_cost",
                "bootstrap_confidence_intervals",
                "distribution_by_task_class",
                "outlier_analysis",
                "per_task_ratios",
                "time_to_terminal_outcome",
            )
            if key in (campaign.get("statistics") or {})
        }
        audit_overhead = campaign.get("audit_overhead") or {}
        vectors_identity = str(campaign.get("vectors_identity") or "")
        coverage = coverage or _behavior_coverage(
            (campaign.get("statistics") or {}).get("distribution_by_task_class") or {}
        )
    else:
        if require_shadow_receipt is not None and cohort != "canary":
            raise QualificationCliError(
                f"{cohort} qualification cannot consume a shadow receipt"
            )
        reasons = [
            f"{cohort}_corpus_unavailable_on_current_tree",
            "honest_nonpromotion",
        ]

    disposition = _disposition(
        pair_count=pair_count,
        minimum_tasks=bound_minimum,
        coverage=coverage,
        quality_safety=quality_safety,
        not_admitted_reason=not_admitted_reason,
    )
    if disposition == "insufficient_evidence" and not reasons:
        reasons = ["pair_count_below_minimum"] if pair_count < bound_minimum else [
            "behavior_coverage_incomplete"
        ]
    if disposition == "safety_or_quality_failed" and not reasons:
        reasons = ["escaped_critical_seed"]
    if disposition not in CLOSED_DISPOSITIONS:
        raise QualificationCliError("qualification disposition is not a closed value")

    results = _results_payload(
        cohort=cohort,
        disposition=disposition,
        pair_count=pair_count,
        minimum_tasks=bound_minimum,
        allow_honest_nonpromotion=allow_honest_nonpromotion,
        allow_not_admitted=allow_not_admitted,
        live=live,
        controls=controls,
        statistics=statistics,
        audit_overhead=audit_overhead,
        reasons=reasons,
        vectors_identity=vectors_identity,
    )
    qualification = _base_receipt(
        cohort=cohort,
        disposition=disposition,
        pair_count=pair_count,
        minimum_tasks=bound_minimum,
        commit=commit,
        tree=tree,
        validator_command=command,
        allow_honest_nonpromotion=allow_honest_nonpromotion,
        allow_not_admitted=allow_not_admitted,
        reasons=reasons,
        quality_safety=quality_safety,
        coverage=coverage,
        controls=controls,
        statistics=statistics,
        audit_overhead=audit_overhead,
        results_identity=str(results.get("identity") or ""),
        live=live,
    )
    if qualification["live"] is True or results["live"] is True:
        raise QualificationCliError("qualification evidence cannot be represented as live")
    if qualification["hermetic_sufficient_for_production_promotion"] is not False:
        raise QualificationCliError("hermetic evidence cannot satisfy production promotion")
    if qualification["production_qualified"] is not False:
        raise QualificationCliError("qualification cannot self-label production_qualified")
    _write_json(Path(output), results)
    _write_json(Path(qualification_output), qualification)
    return {
        "disposition": disposition,
        "pair_count": pair_count,
        "qualification": qualification,
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASEH paired qualification writer")
    parser.add_argument(
        "--cohort",
        required=True,
        choices=tuple(COHORT_TASK),
        help="qualification cohort to emit",
    )
    parser.add_argument(
        "--minimum-tasks",
        type=int,
        help="minimum paired tasks required for evidence_qualified",
    )
    parser.add_argument("--output", type=Path, required=True, help="campaign results JSON")
    parser.add_argument(
        "--qualification-output",
        type=Path,
        required=True,
        help="qualification receipt JSON",
    )
    parser.add_argument(
        "--allow-honest-nonpromotion",
        action="store_true",
        help="exit 0 for insufficient_evidence and safety_or_quality_failed",
    )
    parser.add_argument(
        "--allow-not-admitted",
        action="store_true",
        help="exit 0 for not_admitted (canary zero-mutation)",
    )
    parser.add_argument(
        "--require-shadow-receipt",
        type=Path,
        help="canary prerequisite live-shadow results path",
    )
    parser.add_argument("--write", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    command = [
        "python3",
        "benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py",
        *(list(argv) if argv is not None else sys.argv[1:]),
    ]
    try:
        ran = run_qualification(
            cohort=str(args.cohort),
            output=args.output,
            qualification_output=args.qualification_output,
            minimum_tasks=args.minimum_tasks,
            allow_honest_nonpromotion=bool(args.allow_honest_nonpromotion),
            allow_not_admitted=bool(args.allow_not_admitted),
            require_shadow_receipt=args.require_shadow_receipt,
            validator_command=command,
        )
    except (QualificationCliError, OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    disposition = str(ran["disposition"])
    if disposition == "evidence_qualified":
        return 0
    if args.allow_honest_nonpromotion and disposition in {
        "insufficient_evidence",
        "safety_or_quality_failed",
    }:
        return 0
    if args.allow_not_admitted and disposition == "not_admitted":
        return 0
    sys.stderr.write(f"qualification disposition is {disposition}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
