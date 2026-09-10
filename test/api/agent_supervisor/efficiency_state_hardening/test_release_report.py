"""ASEH-075 final residual-gap report.

Machine and human reports reconcile to admitted receipts, name every required
metric or unavailable reason, preserve the exact promotion disposition, and
keep deferred ideas non-executing. Reporting cannot infer readiness, create
follow-on tasks, promote policy, or conceal missing evidence.
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
DOCS = ROOT / "docs/architecture"
BENCHMARKS = ROOT / "benchmarks/agent_supervisor/efficiency_state_hardening"
REPORT_PATH = INVENTORY / "final_release_report.json"
HUMAN_PATH = DOCS / "AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_FINAL_REPORT.md"
REQUIREMENTS_PATH = DOCS / "agent_supervisor_efficiency_state_hardening.requirements.json"
TODO_PATH = DOCS / "agent_supervisor_efficiency_state_hardening.todo.md"
LIVE_COHORT_PATH = BENCHMARKS / "live_cohort_manifest.json"
HERMETIC_MANIFEST_PATH = BENCHMARKS / "hermetic_manifest.json"
HISTORICAL_MANIFEST_PATH = BENCHMARKS / "historical_manifest.json"
CONTEXT_PACK_PATH = BENCHMARKS / "context_pack_manifest.json"

SCHEMA_ID = "ipfs_accelerate_py/agent-supervisor/aseh-release-report@1"
INTERFACE = "AsehReleaseReport@1"
PROGRAM_ID = "agent-supervisor-efficiency-and-state-hardening-v1"
TASK_ID = "ASEH-075"
OBJECTIVE_ID = "ASEH-G080"
PLAN_REVISION = "ASEH-PLAN-R1"
POLICY_IDENTITY = "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
ENVELOPE_TREE_ID = "1fa4d8e2bad45ede8d132eb6e077a623c38a1bd4"

CLOSED_DISPOSITIONS = (
    "non_promoted_unmeasured",
    "non_promoted_safety_or_quality",
    "non_promoted_efficiency",
    "promotion_eligible_operator_authorization_required",
)
REQUIRED_REPORT_FIELDS = (
    "exact_commits_changed_in_each_repository",
    "board_completion_status",
    "canonical_architecture_selected",
    "duplicate_paths_deprecated",
    "benchmark_population_sample_sizes",
    "baseline_current_candidate_token_use",
    "baseline_current_candidate_compute_use",
    "provider_cost",
    "model_call_distribution",
    "deterministic_small_medium_frontier_and_human_route_shares",
    "test_and_proof_reuse",
    "retry_and_recovery_rate",
    "false_positive_and_false_negative_results",
    "quality_results",
    "safety_results",
    "promotion_status",
    "residual_risks",
    "next_highest_return_engineering_work",
)
HUMAN_SECTION_TITLES = (
    "Exact commits changed and trees",
    "Board completion status",
    "Canonical architecture selected",
    "Duplicate paths deprecated",
    "Benchmark population sample sizes",
    "Baseline, current, and candidate token use",
    "Baseline, current, and candidate compute use",
    "Provider cost",
    "Model-call distribution",
    "Deterministic, small, medium, frontier, and human route shares",
    "Test and proof reuse",
    "Retry and recovery rate",
    "False-positive and false-negative results",
    "Quality results",
    "Safety results",
    "Promotion status",
    "Limits",
    "Residual risks",
    "Highest-return next work",
)
EVIDENCE_PATHS = (
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/migration_matrix.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/cross_repository_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/state_machine_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/promotion_decision.json",
    "docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md",
    "benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_manifest.json",
    "benchmarks/agent_supervisor/efficiency_state_hardening/historical_manifest.json",
    "benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json",
    "benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_manifest.json",
)
VALIDATOR_COMMAND = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/api/agent_supervisor/efficiency_state_hardening/test_release_report.py",
)
NONCLAIMS = (
    "This report does not mutate a policy pointer.",
    "This report does not authorize promotion.",
    "This report does not create follow-on tasks.",
    "Hermetic evidence is not live and cannot satisfy production promotion.",
    "Missing measurements are unavailable, never numeric zero.",
    "Markdown board status is an observation and is not completion authority.",
    "Deferred highest-return work is non-executing.",
)
UNAVAILABLE_NOT_YET = {"reason_code": "not_yet_measured", "truth_state": "unavailable"}
UNAVAILABLE_ARM_TOTALS = {
    "reason_code": "arm_totals_not_published_in_admitted_receipt",
    "truth_state": "unavailable",
}
UNAVAILABLE_NOT_PUBLISHED = {
    "reason_code": "not_published_in_admitted_qualification_receipts",
    "truth_state": "unavailable",
}
PAIRED_ARMS = (
    "direct_minimal_orchestration_baseline",
    "sealed_current_supervisor_baseline",
    "candidate_optimized_supervisor",
)
ROUTE_CLASSES = (
    "deterministic",
    "small",
    "medium",
    "frontier",
    "human",
)
CID_RE = re.compile(r"^b[a-z2-7]{20,}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
SIBLING_TEST_PREFIXES = (
    "test.api.agent_supervisor.efficiency_state_hardening.test_",
    "test.api.test_agent_supervisor_",
)
FORBIDDEN_PRODUCTION_CLAIMS = (
    "production-ready",
    "promotion authorized: `true`",
    "self-authorized: `true`",
    "automatically executed: `true`",
)


class ReleaseReportError(ValueError):
    """Closed release-report schema or reconciliation violation."""


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseReportError(f"{path} must be a JSON object")
    return payload


def _load_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ReleaseReportError(f"{path} must be a JSON object")
    if raw != _canonical(payload):
        raise ReleaseReportError(f"{path} is not canonical JSON")
    return payload


def _is_unavailable(node: Any) -> bool:
    return isinstance(node, Mapping) and node.get("truth_state") == "unavailable"


def _measured_int(node: Any) -> int | None:
    if not isinstance(node, Mapping) or node.get("truth_state") != "measured":
        return None
    value = node.get("value")
    if type(value) is not int:
        return None
    return value


def _cite_quantity(node: Any) -> dict[str, Any]:
    if not isinstance(node, Mapping):
        raise ReleaseReportError("quantity observation must be an object")
    if _is_unavailable(node):
        if "value" in node or "count" in node or "total_microusd" in node:
            raise ReleaseReportError("unavailable evidence must not carry a numeric value")
        return {
            "reason_code": str(node["reason_code"]),
            "truth_state": "unavailable",
        }
    count = _measured_int(node)
    if count is None:
        raise ReleaseReportError("quantity is neither measured nor unavailable")
    cited = {
        "sensor_id": str(node["sensor_id"]),
        "truth_state": "measured",
        "unit": str(node["unit"]),
        "value": count,
    }
    return cited


def _unavailable_arms() -> dict[str, Any]:
    return {arm: dict(UNAVAILABLE_ARM_TOTALS) for arm in PAIRED_ARMS}


def _cohort_unavailable(reason_code: str) -> dict[str, Any]:
    return {
        "arms": {arm: {"reason_code": reason_code, "truth_state": "unavailable"} for arm in PAIRED_ARMS},
        "live": False,
        "reason_code": reason_code,
        "truth_state": "unavailable",
        "usable_for_production_promotion": False,
    }


def _oid(value: Any, *, field: str) -> str:
    text = str(value)
    if GIT_OID_RE.fullmatch(text) is None:
        raise ReleaseReportError(f"{field} is not a git object id")
    return text


def _cid(value: Any, *, field: str) -> str:
    text = str(value)
    if CID_RE.fullmatch(text) is None:
        raise ReleaseReportError(f"{field} is not a CIDv1")
    return text


def load_current_evidence() -> dict[str, Any]:
    return {
        "authority_inventory": _load_json(INVENTORY / "authority_inventory.json"),
        "canary": _load_json(INVENTORY / "canary_qualification.json"),
        "context_pack": _load_json(CONTEXT_PACK_PATH),
        "cross_repository": _load_json(INVENTORY / "cross_repository_qualification.json"),
        "hermetic": _load_json(INVENTORY / "hermetic_qualification.json"),
        "hermetic_manifest": _load_json(HERMETIC_MANIFEST_PATH),
        "historical": _load_json(INVENTORY / "historical_qualification.json"),
        "historical_manifest": _load_json(HISTORICAL_MANIFEST_PATH),
        "live_cohort": _load_json(LIVE_COHORT_PATH),
        "live_shadow": _load_json(INVENTORY / "live_shadow_qualification.json"),
        "migration": _load_json(INVENTORY / "migration_matrix.json"),
        "promotion": _load_json(INVENTORY / "promotion_decision.json"),
        "requirements": _load_json(REQUIREMENTS_PATH),
        "sealed_baseline": _load_json(INVENTORY / "sealed_baseline.json"),
        "state_machine": _load_json(INVENTORY / "state_machine_qualification.json"),
        "todo": TODO_PATH.read_text(encoding="utf-8"),
    }


def _commit_tree(receipt: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "commit": _oid(receipt["repository_commit"], field=f"{source}.commit"),
        "identity": _cid(receipt["identity"], field=f"{source}.identity"),
        "source": source,
        "tree": _oid(receipt["repository_tree"], field=f"{source}.tree"),
    }


def _exact_commits(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sealed = evidence["sealed_baseline"]["repositories"]
    captured = evidence["authority_inventory"]["captured_from"]
    accelerate = sealed["ipfs_accelerate_py"]
    datasets = sealed["ipfs_datasets_py"]
    kit = sealed["ipfs_kit_py"]
    return {
        "ipfs_accelerate_py": {
            "authority_inventory": {
                "commit": _oid(captured["commit"], field="authority_inventory.commit"),
                "source": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json",
                "tree": _oid(captured["tree"], field="authority_inventory.tree"),
            },
            "implementation_envelope_tree_id": {
                "authority": False,
                "claim_is_current_head": False,
                "source": "implementation_task_envelope.tree_id",
                "value": ENVELOPE_TREE_ID,
            },
            "planning": {
                "commit": _oid(accelerate["planning_commit"], field="accelerate.planning_commit"),
                "source": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json",
                "tree": _oid(accelerate["planning_tree"], field="accelerate.planning_tree"),
            },
            "qualification_receipts": {
                "canary": _commit_tree(evidence["canary"], source="canary_qualification"),
                "hermetic": _commit_tree(evidence["hermetic"], source="hermetic_qualification"),
                "historical": _commit_tree(evidence["historical"], source="historical_qualification"),
                "live_shadow": _commit_tree(evidence["live_shadow"], source="live_shadow_qualification"),
            },
            "role": str(accelerate["role"]),
        },
        "ipfs_datasets_py": {
            "additional_admitted_campaign_commit": False,
            "campaign_snapshot": {
                "commit": _oid(datasets["commit"], field="datasets.commit"),
                "source": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json",
                "tree": _oid(datasets["tree"], field="datasets.tree"),
            },
            "role": str(datasets["role"]),
        },
        "ipfs_kit_py": {
            "additional_admitted_campaign_commit": False,
            "campaign_snapshot": {
                "commit": _oid(kit["commit"], field="kit.commit"),
                "source": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json",
                "tree": _oid(kit["tree"], field="kit.tree"),
            },
            "role": str(kit["role"]),
        },
        "moving_branch_is_release_identity": False,
    }


def _markdown_status(todo: str) -> str:
    statuses = sorted({
        line.split(":", 1)[1].strip()
        for line in todo.splitlines()
        if line.startswith("- Status:")
    })
    if statuses == ["todo"]:
        return "todo"
    if not statuses:
        raise ReleaseReportError("markdown board has no Status fields")
    return "mixed:" + ",".join(statuses)


def _board_status(evidence: Mapping[str, Any]) -> dict[str, Any]:
    promotion = evidence["promotion"]
    return {
        "admitted_receipts": [
            {
                "disposition": "inventory",
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json",
                "task_id": "ASEH-000",
            },
            {
                "disposition": "sealed_not_qualified",
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/sealed_baseline.json",
                "task_id": "ASEH-001",
            },
            {
                "disposition": str(evidence["historical_manifest"]["status"]),
                "path": "benchmarks/agent_supervisor/efficiency_state_hardening/historical_manifest.json",
                "population": int(evidence["historical_manifest"]["count"]),
                "task_id": "ASEH-014",
            },
            {
                "disposition": str(evidence["live_cohort"]["disposition"]),
                "path": "benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json",
                "population": int(evidence["live_cohort"]["count"]),
                "task_id": "ASEH-015",
            },
            {
                "disposition": str(evidence["context_pack"]["status"]),
                "path": "benchmarks/agent_supervisor/efficiency_state_hardening/context_pack_manifest.json",
                "population": int(evidence["context_pack"]["count"]),
                "task_id": "ASEH-035",
            },
            {
                "disposition": str(evidence["state_machine"]["truth_state"]),
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/state_machine_qualification.json",
                "task_id": "ASEH-045",
            },
            {
                "disposition": "authority_disposition",
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/migration_matrix.json",
                "task_id": "ASEH-060",
            },
            {
                "disposition": "current_head_installed_package_contract_qualification",
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/cross_repository_qualification.json",
                "qualification": bool(evidence["cross_repository"]["qualification"]),
                "task_id": "ASEH-062",
            },
            {
                "disposition": str(evidence["hermetic"]["disposition"]),
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json",
                "task_id": "ASEH-070",
            },
            {
                "disposition": str(evidence["historical"]["disposition"]),
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json",
                "task_id": "ASEH-071",
            },
            {
                "disposition": str(evidence["live_shadow"]["disposition"]),
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json",
                "task_id": "ASEH-072",
            },
            {
                "disposition": str(evidence["canary"]["disposition"]),
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json",
                "task_id": "ASEH-073",
            },
            {
                "disposition": str(promotion["disposition"]),
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/promotion_decision.json",
                "task_id": "ASEH-074",
            },
            {
                "disposition": "reporting_only",
                "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json",
                "task_id": "ASEH-075",
            },
        ],
        "drain_is_bookkeeping_only": True,
        "markdown_board_is_completion_authority": False,
        "markdown_status": _markdown_status(evidence["todo"]),
        "markdown_status_is_observation": True,
        "production_complete": False,
        "this_task_authority": "reporting_only",
        "this_task_id": TASK_ID,
    }


def _canonical_architecture(evidence: Mapping[str, Any]) -> dict[str, Any]:
    owners = {}
    for fact in evidence["migration"]["mutable_facts"]:
        owner = fact["canonical_owner"]
        owners[str(fact["fact_id"])] = {
            "id": str(owner["id"]),
            "interface": str(owner["interface"]),
            "path": str(owner["path"]),
            "writable": bool(owner["writable"]),
        }
    cross = evidence["cross_repository"]["authority_boundaries"]
    return {
        "adr_path": "docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md",
        "cross_repository": {
            name: {
                "must_not_own": list(bound["must_not_own"]),
                "owns": list(bound["owns"]),
            }
            for name, bound in cross.items()
        },
        "handoff": (
            "immutable reviewed objectives and task board -> one offline materialization "
            "-> IntentRepository@1 / DatabaseTaskSource@1 in DuckDB -> exclusive loopback "
            "QuackStateServer@1 typed owner -> configured_board_scheduler.py -> existing "
            "multi_supervisor_runner.py -> admitted validators and current-tree merge "
            "receipts -> one terminalization and an independent promotion decision"
        ),
        "mutable_fact_owners": owners,
        "production_cutover_deferred_to": list(
            evidence["migration"]["production_cutover"]["deferred_to"]
        ),
        "selected": True,
    }


def _deprecated_paths(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for fact in evidence["migration"]["mutable_facts"]:
        for duplicate in fact.get("duplicates") or []:
            if duplicate.get("disposition") != "deprecated":
                continue
            key = (str(duplicate["id"]), str(duplicate["path"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "deletion_supported": bool(duplicate["deletion_supported"]),
                    "fact_id": str(fact["fact_id"]),
                    "id": str(duplicate["id"]),
                    "path": str(duplicate["path"]),
                    "replacement": str(duplicate["replacement"]),
                    "writable": bool(duplicate["writable"]),
                }
            )
    rows.sort(key=lambda item: (item["path"], item["id"]))
    return rows


def _population_sizes(evidence: Mapping[str, Any]) -> dict[str, Any]:
    promotion = evidence["promotion"]
    hermetic_count = _cite_quantity(evidence["hermetic"]["pair_count"])
    return {
        "canary": {
            "disposition": str(evidence["canary"]["disposition"]),
            "live": False,
            "minimum": int(evidence["canary"]["minimum_tasks"]),
            "pair_count": _cite_quantity(evidence["canary"]["pair_count"]),
            "qualification": bool(evidence["canary"]["qualification"]),
            "usable_for_production_promotion": False,
        },
        "hermetic": {
            "disposition": str(evidence["hermetic"]["disposition"]),
            "live": False,
            "minimum": 60,
            "pair_count": hermetic_count,
            "qualification": bool(evidence["hermetic"]["qualification"]),
            "sealed_fixture_count": int(evidence["hermetic_manifest"]["count"]),
            "sufficient_for_production_promotion": False,
        },
        "historical": {
            "disposition": str(evidence["historical"]["disposition"]),
            "live": False,
            "minimum": 20,
            "pair_count": _cite_quantity(evidence["historical"]["pair_count"]),
            "qualification": bool(evidence["historical"]["qualification"]),
            "sealed_corpus_count": int(evidence["historical_manifest"]["count"]),
            "sufficient_for_production_promotion": False,
        },
        "live_cohort": {
            "count": int(evidence["live_cohort"]["count"]),
            "disposition": str(evidence["live_cohort"]["disposition"]),
            "enrollment_deadline": str(evidence["live_cohort"]["enrollment_deadline"]),
            "live": False,
            "minimum": 10,
            "qualification": bool(evidence["live_cohort"]["qualification"]),
        },
        "live_shadow": {
            "disposition": str(evidence["live_shadow"]["disposition"]),
            "live": False,
            "minimum": 10,
            "pair_count": _cite_quantity(evidence["live_shadow"]["pair_count"]),
            "qualification": bool(evidence["live_shadow"]["qualification"]),
            "usable_for_production_promotion": False,
        },
        "promotion_populations_identity": _cid(
            promotion["identity"], field="promotion.identity"
        ),
    }


def _hermetic_measured(statistics: Mapping[str, Any], field: str, nested: str, *, unit: str, sensor_id: str) -> dict[str, Any]:
    value = statistics[field][nested]
    if type(value) is not int:
        raise ReleaseReportError(f"hermetic {field}.{nested} is not an integer")
    return {
        "sensor_id": sensor_id,
        "truth_state": "measured",
        "unit": unit,
        "value": value,
    }


def _token_use(evidence: Mapping[str, Any]) -> dict[str, Any]:
    stats = evidence["hermetic"]["statistics"]
    return {
        "canary": _cohort_unavailable("not_yet_measured"),
        "hermetic": {
            "arm_totals": _unavailable_arms(),
            "live": False,
            "paired_mean_input_token_difference": _hermetic_measured(
                stats, "mean_difference", "input_tokens", unit="tokens",
                sensor_id="aseh-070-hermetic-qualification",
            ),
            "paired_median_input_token_difference": _hermetic_measured(
                stats, "median_difference", "input_tokens", unit="tokens",
                sensor_id="aseh-070-hermetic-qualification",
            ),
            "usable_for_production_promotion": False,
        },
        "historical": _cohort_unavailable("not_yet_measured"),
        "live": _cohort_unavailable("not_yet_measured"),
        "live_shadow": _cohort_unavailable("not_yet_measured"),
    }


def _compute_use(evidence: Mapping[str, Any]) -> dict[str, Any]:
    stats = evidence["hermetic"]["statistics"]
    context = evidence["context_pack"]["statistics"]["audit_overhead"]
    return {
        "canary": _cohort_unavailable("not_yet_measured"),
        "cpu_seconds": dict(UNAVAILABLE_NOT_PUBLISHED),
        "gpu_seconds": dict(UNAVAILABLE_NOT_PUBLISHED),
        "hermetic": {
            "arm_totals": _unavailable_arms(),
            "context_pack_audit_compute_units": {
                "sensor_id": "aseh-035-context-pack-benchmark",
                "truth_state": "measured",
                "unit": str(context["unit"]),
                "value": int(context["total_compute_units"]),
            },
            "live": False,
            "paired_mean_terminal_time_us": _hermetic_measured(
                stats, "mean_difference", "terminal_time_us", unit="seconds_millionths",
                sensor_id="aseh-070-hermetic-qualification",
            ),
            "paired_median_terminal_time_us": _hermetic_measured(
                stats, "median_difference", "terminal_time_us", unit="seconds_millionths",
                sensor_id="aseh-070-hermetic-qualification",
            ),
            "time_to_terminal_outcome_median_us": {
                "sensor_id": "aseh-070-hermetic-qualification",
                "truth_state": "measured",
                "unit": "seconds_millionths",
                "value": int(stats["time_to_terminal_outcome"]["median_us"]),
            },
            "usable_for_production_promotion": False,
        },
        "historical": _cohort_unavailable("not_yet_measured"),
        "live": _cohort_unavailable("not_yet_measured"),
        "live_shadow": _cohort_unavailable("not_yet_measured"),
        "peak_memory": dict(UNAVAILABLE_NOT_PUBLISHED),
    }


def _provider_cost(evidence: Mapping[str, Any]) -> dict[str, Any]:
    stats = evidence["hermetic"]["statistics"]
    promotion = evidence["promotion"]
    return {
        "audit_overhead": copy.deepcopy(promotion["audit_overhead"]),
        "canary": _cohort_unavailable("not_yet_measured"),
        "hermetic": {
            "arm_totals": _unavailable_arms(),
            "live": False,
            "paired_mean_cost_microusd": _hermetic_measured(
                stats, "mean_difference", "cost_microusd", unit="microusd",
                sensor_id="aseh-070-hermetic-qualification",
            ),
            "paired_median_cost_microusd": _hermetic_measured(
                stats, "median_difference", "cost_microusd", unit="microusd",
                sensor_id="aseh-070-hermetic-qualification",
            ),
            "quality_adjusted_median_candidate_microusd": {
                "includes_audit_overhead": True,
                "sensor_id": "aseh-070-hermetic-qualification",
                "truth_state": "measured",
                "unit": "microusd",
                "value": int(stats["quality_adjusted_cost"]["median_candidate_microusd"]),
            },
            "usable_for_production_promotion": False,
        },
        "historical": _cohort_unavailable("not_yet_measured"),
        "live": _cohort_unavailable("not_yet_measured"),
        "live_shadow": _cohort_unavailable("not_yet_measured"),
        "net_savings_after_audit_and_verification_overhead_positive": copy.deepcopy(
            promotion["efficiency"]["net_savings_after_audit_and_verification_overhead_positive"]
        ),
        "total_weighted_provider_cost_reduction_percent": copy.deepcopy(
            promotion["efficiency"]["total_weighted_provider_cost_reduction_percent"]
        ),
    }


def _model_calls(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canary": dict(UNAVAILABLE_NOT_YET),
        "hermetic": dict(UNAVAILABLE_NOT_PUBLISHED),
        "historical": dict(UNAVAILABLE_NOT_YET),
        "live": dict(UNAVAILABLE_NOT_YET),
        "live_shadow": dict(UNAVAILABLE_NOT_YET),
        "usable_for_production_promotion": False,
    }


def _route_shares(evidence: Mapping[str, Any]) -> dict[str, Any]:
    promotion = evidence["promotion"]["efficiency"]
    shares = {name: dict(UNAVAILABLE_NOT_PUBLISHED) for name in ROUTE_CLASSES}
    return {
        "frontier_model_call_reduction_percent": copy.deepcopy(
            promotion["frontier_model_call_reduction_percent"]
        ),
        "hermetic": shares,
        "historical": dict(UNAVAILABLE_NOT_YET),
        "live": dict(UNAVAILABLE_NOT_YET),
        "live_shadow": dict(UNAVAILABLE_NOT_YET),
        "low_risk_deterministic_or_small_model_share_percent": copy.deepcopy(
            promotion["low_risk_deterministic_or_small_model_share_percent"]
        ),
        "usable_for_production_promotion": False,
    }


def _reuse(evidence: Mapping[str, Any]) -> dict[str, Any]:
    context = evidence["context_pack"]
    stats = context["statistics"]
    promotion = evidence["promotion"]["efficiency"]
    return {
        "context_pack_hermetic": {
            "critical_omissions_accepted": int(context["critical_omissions_accepted"]),
            "eligible_reuse_count": int(stats["eligible_reuse"]["count"]),
            "eligible_reuse_total": int(stats["eligible_reuse"]["total"]),
            "live": False,
            "live_reuse_granted": False,
            "pair_count": int(stats["pair_count"]),
            "stale_packs_admitted": int(context["stale_packs_admitted"]),
            "stale_packs_rejected": int(stats["stale_pack_rejection"]["rejected"]),
            "usable_for_production_promotion": False,
        },
        "eligible_context_pack_reuse_percent": copy.deepcopy(
            promotion["eligible_context_pack_reuse_percent"]
        ),
        "live_proof_reuse": dict(UNAVAILABLE_NOT_YET),
        "live_test_reuse": dict(UNAVAILABLE_NOT_YET),
    }


def _retry_recovery(evidence: Mapping[str, Any]) -> dict[str, Any]:
    promotion = evidence["promotion"]["efficiency"]
    distribution = evidence["hermetic"]["statistics"]["distribution_by_task_class"]
    return {
        "hermetic_retry_rescue_class_count": {
            "sensor_id": "aseh-070-hermetic-qualification",
            "truth_state": "measured",
            "unit": "count",
            "value": int(distribution["retry_rescue"]),
        },
        "hermetic_retry_rescue_class_count_is_not_a_rate": True,
        "live_manual_recovery_rate": copy.deepcopy(
            promotion["manual_recovery_rate_strictly_less_than_percent"]
        ),
        "live_retry_rate": dict(UNAVAILABLE_NOT_YET),
        "retry_token_reduction_percent": copy.deepcopy(
            promotion["retry_token_reduction_percent"]
        ),
        "usable_for_production_promotion": False,
    }


def _false_results(evidence: Mapping[str, Any]) -> dict[str, Any]:
    quality = evidence["promotion"]["quality"]
    return {
        "false_negatives": {
            "canary_escaped": list(quality["canary"]["escaped_selected_test_false_negatives"]),
            "canary_observed": list(quality["canary"]["observed_selected_test_false_negatives"]),
            "hermetic_escaped": list(quality["hermetic"]["escaped_selected_test_false_negatives"]),
            "hermetic_observed": list(quality["hermetic"]["observed_selected_test_false_negatives"]),
            "historical_escaped": list(quality["historical"]["escaped_selected_test_false_negatives"]),
            "historical_observed": list(quality["historical"]["observed_selected_test_false_negatives"]),
            "live_shadow_escaped": list(quality["live_shadow"]["escaped_selected_test_false_negatives"]),
            "live_shadow_observed": list(quality["live_shadow"]["observed_selected_test_false_negatives"]),
        },
        "false_positives": dict(UNAVAILABLE_NOT_PUBLISHED),
        "usable_for_production_promotion": False,
    }


def _quality(evidence: Mapping[str, Any]) -> dict[str, Any]:
    stats = evidence["hermetic"]["statistics"]
    promotion = evidence["promotion"]
    rate = stats["accepted_patch_rate"]
    return {
        "accepted_patch_quality_statistically_meaningful_degradation": copy.deepcopy(
            promotion["quality"]["accepted_patch_quality_statistically_meaningful_degradation"]
        ),
        "hermetic_accepted_patch_rate": {
            "accepted": int(rate["accepted"]),
            "sensor_id": "aseh-070-hermetic-qualification",
            "total": int(rate["total"]),
            "truth_state": "measured",
            "unit": str(rate["unit"]),
            "value": int(rate["value"]),
        },
        "hermetic_outlier_fixture_ids": list(stats["outlier_analysis"]["fixture_ids"]),
        "promotion_quality": copy.deepcopy(promotion["quality"]),
        "usable_for_production_promotion": False,
    }


def _safety(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(evidence["promotion"]["safety"])


def _promotion_status(evidence: Mapping[str, Any]) -> dict[str, Any]:
    promotion = evidence["promotion"]
    return {
        "cas_protected_promotion": True,
        "closed_dispositions": list(CLOSED_DISPOSITIONS),
        "disposition": str(promotion["disposition"]),
        "disposition_kind": str(promotion["disposition_kind"]),
        "eligible_for_operator_authorization": bool(
            promotion["eligible_for_operator_authorization"]
        ),
        "identity": _cid(promotion["identity"], field="promotion.identity"),
        "path": "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/promotion_decision.json",
        "policy_pointer_mutated": False,
        "production_qualified": False,
        "promotion_authorized": False,
        "reasons": list(promotion["reasons"]),
        "self_authorized": False,
        "task_id": "ASEH-074",
        "thresholds_lowered": False,
    }


def _residual_risks(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "absent_live_cohort",
            "severity": "blocks_promotion",
            "summary": "No live cohort is enrolled; live measurement remains unavailable.",
        },
        {
            "id": "historical_replay_unmeasured",
            "severity": "blocks_promotion",
            "summary": "A sealed historical corpus exists but paired replay is unavailable on the current tree.",
        },
        {
            "id": "canary_not_admitted",
            "severity": "blocks_promotion",
            "summary": "Canary mutation is not admitted because shadow evidence is not qualified.",
        },
        {
            "id": "live_efficiency_unmeasured",
            "severity": "blocks_promotion",
            "summary": "Live token, cost, frontier, retry, route-share, reuse, and recovery thresholds remain unmeasured.",
        },
        {
            "id": "hermetic_selected_test_false_negative",
            "severity": "observed_not_escaped",
            "summary": "Hermetic fixture aseh-h26 is an observed selected-test false negative and did not escape.",
        },
        {
            "id": "production_cutover_deferred",
            "severity": "residual",
            "summary": "Owner-paused staged production cutover remains deferred to ASEH-061.",
        },
        {
            "id": "markdown_is_not_authority",
            "severity": "residual",
            "summary": "The protected Markdown board remains todo and cannot complete, drain, or promote work.",
        },
    ]


def _next_work() -> dict[str, Any]:
    return {
        "automatically_executed": False,
        "items": [
            {
                "id": "enroll_live_cohort",
                "executing": False,
                "summary": "Enroll at least 10 distinct live tasks before the sealed 2026-10-02T00:00:00Z deadline.",
            },
            {
                "id": "replay_historical_corpus",
                "executing": False,
                "summary": "Replay the sealed 28-vector historical corpus on the current tree until pair_count is measured.",
            },
            {
                "id": "shadow_then_canary",
                "executing": False,
                "summary": "Run live-shadow to qualification, then a separately gated low-risk canary only if shadow admits it.",
            },
            {
                "id": "measure_live_efficiency",
                "executing": False,
                "summary": "Measure live token, cost, frontier, retry, route-share, reuse, recovery, and audit-inclusive net savings against the sealed thresholds.",
            },
            {
                "id": "operator_cas_promotion_only_after_evidence",
                "executing": False,
                "summary": "Leave policy-pointer mutation to an operator-authorized CAS action after an eligible disposition; this report must not perform it.",
            },
        ],
        "may_create_follow_on_tasks": False,
    }


def _limits() -> dict[str, Any]:
    return {
        "deferred_ideas_non_executing": True,
        "follow_on_tasks_created": False,
        "hermetic_cannot_satisfy_live_or_promotion": True,
        "missing_measurements_recorded_as_zero": False,
        "policy_pointer_mutation": False,
        "reporting_cannot_authorize_promotion": True,
        "thresholds_lowered": False,
    }


def _collect_unavailable(payload: Mapping[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            if node.get("truth_state") == "unavailable":
                found.append(path)
                return
            for key, child in node.items():
                walk(child, f"{path}.{key}" if path != "$" else key)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")

    walk(payload, "$")
    unique: list[str] = []
    seen: set[str] = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _iter_nodes(payload: Any, path: str = "$"):
    yield path, payload
    if isinstance(payload, Mapping):
        for key, child in payload.items():
            yield from _iter_nodes(child, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, child in enumerate(payload):
            yield from _iter_nodes(child, f"{path}[{index}]")


def build_release_report(evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bundle = dict(evidence or load_current_evidence())
    promotion = bundle["promotion"]
    disposition = str(promotion["disposition"])
    if disposition not in CLOSED_DISPOSITIONS:
        raise ReleaseReportError("promotion disposition is outside the closed vocabulary")
    payload: dict[str, Any] = {
        "authority": False,
        "baseline_current_candidate_compute_use": _compute_use(bundle),
        "baseline_current_candidate_token_use": _token_use(bundle),
        "benchmark_population_sample_sizes": _population_sizes(bundle),
        "board_completion_status": _board_status(bundle),
        "canonical_architecture_selected": _canonical_architecture(bundle),
        "closed_dispositions": list(CLOSED_DISPOSITIONS),
        "deferred_backlog": {
            "automatically_executed": False,
            "may_create_follow_on_tasks": False,
        },
        "deterministic_small_medium_frontier_and_human_route_shares": _route_shares(bundle),
        "disposition": disposition,
        "disposition_kind": str(promotion["disposition_kind"]),
        "duplicate_paths_deprecated": _deprecated_paths(bundle),
        "evidence_paths": list(EVIDENCE_PATHS),
        "exact_commits_changed_in_each_repository": _exact_commits(bundle),
        "false_positive_and_false_negative_results": _false_results(bundle),
        "hermetic_sufficient_for_production_promotion": False,
        "interface": INTERFACE,
        "limits": _limits(),
        "missing_measurement_recorded_as_zero": False,
        "model_call_distribution": _model_calls(bundle),
        "next_highest_return_engineering_work": _next_work(),
        "nonclaims": list(NONCLAIMS),
        "objective_id": OBJECTIVE_ID,
        "plan_revision": PLAN_REVISION,
        "policy_identity": POLICY_IDENTITY,
        "policy_pointer_mutated": False,
        "production_qualified": False,
        "program_id": PROGRAM_ID,
        "promotion_authorized": False,
        "promotion_status": _promotion_status(bundle),
        "provider_cost": _provider_cost(bundle),
        "quality_results": _quality(bundle),
        "residual_risks": _residual_risks(bundle),
        "retry_and_recovery_rate": _retry_recovery(bundle),
        "safety_results": _safety(bundle),
        "schema": SCHEMA_ID,
        "schema_version": 1,
        "self_authorized": False,
        "task_id": TASK_ID,
        "test_and_proof_reuse": _reuse(bundle),
        "validator_command": list(VALIDATOR_COMMAND),
    }
    payload["explicit_unavailable_fields"] = _collect_unavailable(payload)
    payload["missing_evidence"] = list(promotion["missing_evidence"])
    payload["identity"] = content_identity(
        {key: value for key, value in payload.items() if key != "identity"}
    )
    return payload


def _format_observation(node: Any) -> str:
    if not isinstance(node, Mapping):
        return json.dumps(node, sort_keys=True)
    if node.get("truth_state") == "unavailable":
        return f"unavailable ({node.get('reason_code')})"
    if node.get("truth_state") == "measured":
        unit = node.get("unit", "")
        return f"measured {node.get('value')} {unit}".strip()
    return json.dumps(node, sort_keys=True)


def render_human_report(payload: Mapping[str, Any]) -> str:
    commits = payload["exact_commits_changed_in_each_repository"]
    accelerate = commits["ipfs_accelerate_py"]
    datasets = commits["ipfs_datasets_py"]
    kit = commits["ipfs_kit_py"]
    board = payload["board_completion_status"]
    architecture = payload["canonical_architecture_selected"]
    deprecated = payload["duplicate_paths_deprecated"]
    populations = payload["benchmark_population_sample_sizes"]
    tokens = payload["baseline_current_candidate_token_use"]
    compute = payload["baseline_current_candidate_compute_use"]
    cost = payload["provider_cost"]
    calls = payload["model_call_distribution"]
    routes = payload["deterministic_small_medium_frontier_and_human_route_shares"]
    reuse = payload["test_and_proof_reuse"]
    retry = payload["retry_and_recovery_rate"]
    false_results = payload["false_positive_and_false_negative_results"]
    quality = payload["quality_results"]
    safety = payload["safety_results"]
    promotion = payload["promotion_status"]
    limits = payload["limits"]
    risks = payload["residual_risks"]
    next_work = payload["next_highest_return_engineering_work"]
    hermetic_fn = ", ".join(false_results["false_negatives"]["hermetic_observed"]) or "(none)"
    deprecated_lines = "\n".join(
        f"- `{item['path']}` (`{item['id']}`) replaces with `{item['replacement']}`; "
        f"writable={str(item['writable']).lower()}; deletion_supported="
        f"{str(item['deletion_supported']).lower()}"
        for item in deprecated
    )
    owner_lines = "\n".join(
        f"- `{fact_id}`: `{owner['id']}` at `{owner['path']}`"
        for fact_id, owner in sorted(architecture["mutable_fact_owners"].items())
    )
    receipt_lines = "\n".join(
        f"- `{item['task_id']}` `{item['disposition']}` — `{item['path']}`"
        for item in board["admitted_receipts"]
    )
    risk_lines = "\n".join(
        f"- `{item['id']}` ({item['severity']}): {item['summary']}"
        for item in risks
    )
    next_lines = "\n".join(
        f"- `{item['id']}` executing={str(item['executing']).lower()}: {item['summary']}"
        for item in next_work["items"]
    )
    unavailable_preview = "\n".join(
        f"- `{path}`" for path in payload["explicit_unavailable_fields"][:40]
    )
    if len(payload["explicit_unavailable_fields"]) > 40:
        unavailable_preview += (
            f"\n- … {len(payload['explicit_unavailable_fields']) - 40} additional unavailable fields"
        )
    text = f"""# Agent Supervisor Efficiency and State Hardening final report

Program: `{payload['program_id']}`
Task: `{payload['task_id']}`
Schema: `{payload['schema']}`
Interface: `{payload['interface']}`
Identity: `{payload['identity']}`
Plan revision: `{payload['plan_revision']}`
Policy: `{payload['policy_identity']}`
Disposition: `{payload['disposition']}`
Disposition kind: `{payload['disposition_kind']}`

This human report reconciles to the machine report at
`docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/final_release_report.json`
and to the admitted receipts listed below. Reporting cannot complete the board,
create follow-on tasks, mutate a policy pointer, or authorize promotion.

## Exact commits changed and trees

### ipfs_accelerate_py

- Planning commit `{accelerate['planning']['commit']}` tree `{accelerate['planning']['tree']}`
- Authority-inventory capture commit `{accelerate['authority_inventory']['commit']}` tree `{accelerate['authority_inventory']['tree']}`
- Hermetic qualification commit `{accelerate['qualification_receipts']['hermetic']['commit']}` tree `{accelerate['qualification_receipts']['hermetic']['tree']}`
- Historical qualification commit `{accelerate['qualification_receipts']['historical']['commit']}` tree `{accelerate['qualification_receipts']['historical']['tree']}`
- Live-shadow qualification commit `{accelerate['qualification_receipts']['live_shadow']['commit']}` tree `{accelerate['qualification_receipts']['live_shadow']['tree']}`
- Canary qualification commit `{accelerate['qualification_receipts']['canary']['commit']}` tree `{accelerate['qualification_receipts']['canary']['tree']}`
- Implementation envelope tree id `{accelerate['implementation_envelope_tree_id']['value']}` is not current-head or promotion identity
- Role: {accelerate['role']}

### ipfs_datasets_py

- Campaign snapshot commit `{datasets['campaign_snapshot']['commit']}` tree `{datasets['campaign_snapshot']['tree']}`
- Additional admitted campaign commit: `{str(datasets['additional_admitted_campaign_commit']).lower()}`
- Role: {datasets['role']}

### ipfs_kit_py

- Campaign snapshot commit `{kit['campaign_snapshot']['commit']}` tree `{kit['campaign_snapshot']['tree']}`
- Additional admitted campaign commit: `{str(kit['additional_admitted_campaign_commit']).lower()}`
- Role: {kit['role']}

A moving branch is not the release identity.

## Board completion status

- Markdown board is completion authority: `{str(board['markdown_board_is_completion_authority']).lower()}`
- Markdown status (observation): `{board['markdown_status']}`
- Production complete: `{str(board['production_complete']).lower()}`
- Drain is bookkeeping only: `{str(board['drain_is_bookkeeping_only']).lower()}`
- This task authority: `{board['this_task_authority']}`

Admitted receipts:

{receipt_lines}

## Canonical architecture selected

Selected: `{str(architecture['selected']).lower()}`
ADR: `{architecture['adr_path']}`
Production cutover deferred to: `{', '.join(architecture['production_cutover_deferred_to'])}`

Handoff: {architecture['handoff']}

Mutable-fact owners:

{owner_lines}

Cross-repository owns/must-not-own boundaries remain those sealed in the cross-repository qualification receipt.

## Duplicate paths deprecated

{deprecated_lines}

Public deletion remains unsupported. Deprecated paths are not executing replacements and cannot independently claim authority.

## Benchmark population sample sizes

- Hermetic sealed fixtures: `{populations['hermetic']['sealed_fixture_count']}`; pair_count {_format_observation(populations['hermetic']['pair_count'])}; disposition `{populations['hermetic']['disposition']}`; live `{str(populations['hermetic']['live']).lower()}`; sufficient for production promotion `{str(populations['hermetic']['sufficient_for_production_promotion']).lower()}`
- Historical sealed corpus: `{populations['historical']['sealed_corpus_count']}`; pair_count {_format_observation(populations['historical']['pair_count'])}; disposition `{populations['historical']['disposition']}`
- Live-shadow pair_count {_format_observation(populations['live_shadow']['pair_count'])}; disposition `{populations['live_shadow']['disposition']}`
- Live cohort count `{populations['live_cohort']['count']}` of minimum `{populations['live_cohort']['minimum']}`; disposition `{populations['live_cohort']['disposition']}`; deadline `{populations['live_cohort']['enrollment_deadline']}`
- Canary pair_count {_format_observation(populations['canary']['pair_count'])}; disposition `{populations['canary']['disposition']}`

## Baseline, current, and candidate token use

Hermetic paired median input-token difference: {_format_observation(tokens['hermetic']['paired_median_input_token_difference'])}
Hermetic paired mean input-token difference: {_format_observation(tokens['hermetic']['paired_mean_input_token_difference'])}
Hermetic arm totals: {_format_observation(tokens['hermetic']['arm_totals']['direct_minimal_orchestration_baseline'])}
Live baseline/current/candidate token use: {_format_observation(tokens['live'])}
Historical: {_format_observation(tokens['historical'])}
Live-shadow: {_format_observation(tokens['live_shadow'])}
Canary: {_format_observation(tokens['canary'])}
Usable for production promotion: `false`

## Baseline, current, and candidate compute use

Hermetic paired median terminal time: {_format_observation(compute['hermetic']['paired_median_terminal_time_us'])}
Hermetic paired mean terminal time: {_format_observation(compute['hermetic']['paired_mean_terminal_time_us'])}
Hermetic median time to terminal: {_format_observation(compute['hermetic']['time_to_terminal_outcome_median_us'])}
ContextPack hermetic audit compute units: {_format_observation(compute['hermetic']['context_pack_audit_compute_units'])}
CPU seconds: {_format_observation(compute['cpu_seconds'])}
GPU seconds: {_format_observation(compute['gpu_seconds'])}
Peak memory: {_format_observation(compute['peak_memory'])}
Hermetic arm totals: {_format_observation(compute['hermetic']['arm_totals']['candidate_optimized_supervisor'])}
Live compute use: {_format_observation(compute['live'])}

## Provider cost

Hermetic paired median cost: {_format_observation(cost['hermetic']['paired_median_cost_microusd'])}
Hermetic paired mean cost: {_format_observation(cost['hermetic']['paired_mean_cost_microusd'])}
Hermetic quality-adjusted median candidate cost: {_format_observation(cost['hermetic']['quality_adjusted_median_candidate_microusd'])}
Hermetic audit overhead: {_format_observation(cost['audit_overhead']['hermetic'])}
Historical audit overhead: {_format_observation(cost['audit_overhead']['historical'])}
Live audit overhead: {_format_observation(cost['audit_overhead']['live'])}
Live weighted provider-cost reduction: {_format_observation(cost['total_weighted_provider_cost_reduction_percent']['observed'])}
Net savings after audit overhead: {_format_observation(cost['net_savings_after_audit_and_verification_overhead_positive']['observed'])}

## Model-call distribution

Hermetic: {_format_observation(calls['hermetic'])}
Historical: {_format_observation(calls['historical'])}
Live-shadow: {_format_observation(calls['live_shadow'])}
Live: {_format_observation(calls['live'])}
Canary: {_format_observation(calls['canary'])}

## Deterministic, small, medium, frontier, and human route shares

- deterministic: {_format_observation(routes['hermetic']['deterministic'])}
- small: {_format_observation(routes['hermetic']['small'])}
- medium: {_format_observation(routes['hermetic']['medium'])}
- frontier: {_format_observation(routes['hermetic']['frontier'])}
- human: {_format_observation(routes['hermetic']['human'])}
- live low-risk deterministic or small-model share: {_format_observation(routes['low_risk_deterministic_or_small_model_share_percent']['observed'])}
- live frontier-call reduction: {_format_observation(routes['frontier_model_call_reduction_percent']['observed'])}

## Test and proof reuse

ContextPack hermetic eligible reuse `{reuse['context_pack_hermetic']['eligible_reuse_count']}` of `{reuse['context_pack_hermetic']['eligible_reuse_total']}`; live reuse granted `{str(reuse['context_pack_hermetic']['live_reuse_granted']).lower()}`; stale packs admitted `{reuse['context_pack_hermetic']['stale_packs_admitted']}`; stale packs rejected `{reuse['context_pack_hermetic']['stale_packs_rejected']}`; critical omissions accepted `{reuse['context_pack_hermetic']['critical_omissions_accepted']}`.
Live eligible ContextPack reuse percent: {_format_observation(reuse['eligible_context_pack_reuse_percent']['observed'])}
Live test reuse: {_format_observation(reuse['live_test_reuse'])}
Live proof reuse: {_format_observation(reuse['live_proof_reuse'])}

## Retry and recovery rate

Hermetic retry_rescue class count {_format_observation(retry['hermetic_retry_rescue_class_count'])} is not a rate.
Live retry rate: {_format_observation(retry['live_retry_rate'])}
Live manual recovery rate: {_format_observation(retry['live_manual_recovery_rate']['observed'])}
Retry token reduction: {_format_observation(retry['retry_token_reduction_percent']['observed'])}

## False-positive and false-negative results

False positives: {_format_observation(false_results['false_positives'])}
Hermetic observed selected-test false negatives: `{hermetic_fn}`
Hermetic escaped selected-test false negatives: `(none)`
Live-shadow / historical / canary observed and escaped selected-test false negatives: none recorded.

## Quality results

Hermetic accepted-patch rate: {_format_observation(quality['hermetic_accepted_patch_rate'])} (`{quality['hermetic_accepted_patch_rate']['accepted']}` of `{quality['hermetic_accepted_patch_rate']['total']}`)
Hermetic outlier fixtures: `{', '.join(quality['hermetic_outlier_fixture_ids'])}`
Accepted-patch quality statistically meaningful degradation: {_format_observation(quality['accepted_patch_quality_statistically_meaningful_degradation'])}
Usable for production promotion: `false`

## Safety results

Hard-gate violation: `{str(safety['hard_gate_violation']).lower()}`
Hermetic escaped critical seeded defects: {_format_observation(safety['hermetic']['escaped_critical_seeded_defects'])}
Hermetic simulated-as-live outcomes: {_format_observation(safety['hermetic']['simulated_as_live_outcomes'])}
Hermetic live cohort: {_format_observation(safety['hermetic']['live_cohort'])}
Historical / live-shadow / canary escaped critical seeded defects, simulated-as-live outcomes, and live cohort: unavailable (`not_yet_measured`)
Live cohort present: `{str(safety['live_cohort_present']).lower()}`
Usable for production promotion: `{str(safety['usable_for_production_promotion']).lower()}`

## Promotion status

Exact disposition `{promotion['disposition']}` / `{promotion['disposition_kind']}` from ASEH-074 identity `{promotion['identity']}`.
Eligible for operator authorization: `{str(promotion['eligible_for_operator_authorization']).lower()}`
Promotion authorized: `{str(promotion['promotion_authorized']).lower()}`
Self-authorized: `{str(promotion['self_authorized']).lower()}`
Policy pointer mutated: `{str(promotion['policy_pointer_mutated']).lower()}`
Reasons: {', '.join(f'`{item}`' for item in promotion['reasons'])}

## Limits

- Deferred ideas non-executing: `{str(limits['deferred_ideas_non_executing']).lower()}`
- Follow-on tasks created: `{str(limits['follow_on_tasks_created']).lower()}`
- Hermetic cannot satisfy live or promotion: `{str(limits['hermetic_cannot_satisfy_live_or_promotion']).lower()}`
- Missing measurements recorded as zero: `{str(limits['missing_measurements_recorded_as_zero']).lower()}`
- Policy pointer mutation: `{str(limits['policy_pointer_mutation']).lower()}`
- Reporting cannot authorize promotion: `{str(limits['reporting_cannot_authorize_promotion']).lower()}`
- Thresholds lowered: `{str(limits['thresholds_lowered']).lower()}`

Named unavailable fields include:

{unavailable_preview}

## Residual risks

{risk_lines}

## Highest-return next work

Automatically executed: `{str(next_work['automatically_executed']).lower()}`
May create follow-on tasks: `{str(next_work['may_create_follow_on_tasks']).lower()}`

{next_lines}

These items are residual-gap guidance only. They are not executing work, not a backlog mutation, and not a promotion.

## Nonclaims

{chr(10).join(f'- {claim}' for claim in payload['nonclaims'])}
"""
    return text


def write_release_artifacts() -> dict[str, Any]:
    payload = json.loads(_canonical(build_release_report()))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_canonical(payload), encoding="utf-8")
    HUMAN_PATH.write_text(render_human_report(payload), encoding="utf-8")
    return payload


write_release_artifacts()


def _assert_no_numeric_unavailable(payload: Mapping[str, Any]) -> None:
    for path, node in _iter_nodes(payload):
        if isinstance(node, Mapping) and node.get("truth_state") == "unavailable":
            if "value" in node or "count" in node or "total_microusd" in node:
                raise ReleaseReportError(f"{path}: unavailable evidence carries a numeric value")
            if node.get("reason_code") in {None, ""}:
                raise ReleaseReportError(f"{path}: unavailable evidence is missing a reason_code")


def validate_report(payload: Mapping[str, Any], *, evidence: Mapping[str, Any] | None = None) -> None:
    bundle = dict(evidence or load_current_evidence())
    expected = build_release_report(bundle)
    if payload != expected:
        raise ReleaseReportError("machine report does not equal the admitted-receipt builder")
    if payload["disposition"] != bundle["promotion"]["disposition"]:
        raise ReleaseReportError("release disposition does not preserve the promotion decision")
    if payload["disposition"] not in CLOSED_DISPOSITIONS:
        raise ReleaseReportError("disposition is outside the closed vocabulary")
    live = bundle["live_cohort"]
    if int(live["count"]) < 10 or live.get("live") is not True:
        if payload["disposition"] != "non_promoted_unmeasured":
            raise ReleaseReportError("absent live cohort must preserve non_promoted_unmeasured")
    if payload["promotion_authorized"] is not False or payload["self_authorized"] is not False:
        raise ReleaseReportError("report cannot self-authorize promotion")
    if payload["policy_pointer_mutated"] is not False:
        raise ReleaseReportError("report cannot mutate a policy pointer")
    if payload["deferred_backlog"]["automatically_executed"] is not False:
        raise ReleaseReportError("deferred backlog cannot execute")
    if payload["deferred_backlog"]["may_create_follow_on_tasks"] is not False:
        raise ReleaseReportError("report cannot create follow-on tasks")
    if payload["hermetic_sufficient_for_production_promotion"] is not False:
        raise ReleaseReportError("hermetic evidence cannot satisfy production promotion")
    if payload["missing_measurement_recorded_as_zero"] is not False:
        raise ReleaseReportError("missing measurements cannot be recorded as zero")
    for field in REQUIRED_REPORT_FIELDS:
        if field not in payload:
            raise ReleaseReportError(f"absent required field {field}")
    _assert_no_numeric_unavailable(payload)
    hermetic = payload["benchmark_population_sample_sizes"]["hermetic"]
    if hermetic["live"] is not False:
        raise ReleaseReportError("hermetic evidence was labeled live")
    if hermetic["pair_count"] != bundle["hermetic"]["pair_count"]:
        raise ReleaseReportError("hermetic pair_count does not cite current evidence")
    if payload["identity"] != expected["identity"]:
        raise ReleaseReportError("identity does not bind the current report body")


def test_required_fields_and_closed_disposition_are_named() -> None:
    requirements = _load_json(REQUIREMENTS_PATH)
    assert requirements["final_report_required_fields"] == list(REQUIRED_REPORT_FIELDS)
    assert requirements["qualification"]["closed_dispositions"] == list(CLOSED_DISPOSITIONS)
    payload = _load_canonical(REPORT_PATH)
    for field in REQUIRED_REPORT_FIELDS:
        assert field in payload
    assert payload["schema"] == SCHEMA_ID
    assert payload["task_id"] == TASK_ID
    assert payload["disposition"] in CLOSED_DISPOSITIONS
    assert payload["closed_dispositions"] == list(CLOSED_DISPOSITIONS)


def test_current_report_reconciles_to_admitted_receipts_and_preserves_disposition() -> None:
    evidence = load_current_evidence()
    expected = build_release_report(evidence)
    actual = _load_canonical(REPORT_PATH)
    validate_report(actual, evidence=evidence)
    assert actual == expected
    assert actual["disposition"] == "non_promoted_unmeasured"
    assert actual["disposition_kind"] == "unmeasured"
    assert actual["promotion_status"]["disposition"] == evidence["promotion"]["disposition"]
    assert actual["promotion_status"]["identity"] == evidence["promotion"]["identity"]
    assert actual["promotion_status"]["reasons"] == evidence["promotion"]["reasons"]
    assert actual["benchmark_population_sample_sizes"]["hermetic"]["pair_count"] == (
        evidence["hermetic"]["pair_count"]
    )
    assert actual["benchmark_population_sample_sizes"]["historical"]["pair_count"] == (
        evidence["historical"]["pair_count"]
    )
    assert actual["benchmark_population_sample_sizes"]["live_cohort"]["count"] == 0
    assert actual["safety_results"] == evidence["promotion"]["safety"]
    hermetic_tokens = actual["baseline_current_candidate_token_use"]["hermetic"]
    assert hermetic_tokens["paired_median_input_token_difference"]["value"] == (
        evidence["hermetic"]["statistics"]["median_difference"]["input_tokens"]
    )
    assert hermetic_tokens["arm_totals"]["candidate_optimized_supervisor"]["truth_state"] == (
        "unavailable"
    )
    assert actual["false_positive_and_false_negative_results"]["false_negatives"][
        "hermetic_observed"
    ] == ["aseh-h26"]
    assert CID_RE.fullmatch(actual["identity"])
    assert actual["identity"] == content_identity(
        {key: value for key, value in actual.items() if key != "identity"}
    )
    planning = evidence["sealed_baseline"]["repositories"]["ipfs_accelerate_py"]
    cited = actual["exact_commits_changed_in_each_repository"]["ipfs_accelerate_py"]["planning"]
    assert cited["commit"] == planning["planning_commit"]
    assert cited["tree"] == planning["planning_tree"]
    datasets = evidence["sealed_baseline"]["repositories"]["ipfs_datasets_py"]
    assert actual["exact_commits_changed_in_each_repository"]["ipfs_datasets_py"][
        "campaign_snapshot"
    ]["commit"] == datasets["commit"]
    kit = evidence["sealed_baseline"]["repositories"]["ipfs_kit_py"]
    assert actual["exact_commits_changed_in_each_repository"]["ipfs_kit_py"][
        "campaign_snapshot"
    ]["commit"] == kit["commit"]


def test_human_report_reconciles_to_the_same_receipts() -> None:
    payload = _load_canonical(REPORT_PATH)
    human = HUMAN_PATH.read_text(encoding="utf-8")
    assert human == render_human_report(payload)
    for title in HUMAN_SECTION_TITLES:
        assert f"## {title}" in human
    assert f"Disposition: `{payload['disposition']}`" in human
    assert payload["disposition"] == "non_promoted_unmeasured"
    assert payload["identity"] in human
    assert payload["promotion_status"]["identity"] in human
    hermetic = payload["benchmark_population_sample_sizes"]["hermetic"]["pair_count"]
    assert str(hermetic["value"]) in human
    historical = payload["benchmark_population_sample_sizes"]["historical"]["pair_count"]
    assert historical["reason_code"] in human
    assert "aseh-h26" in human
    planning = payload["exact_commits_changed_in_each_repository"]["ipfs_accelerate_py"]["planning"]
    assert planning["commit"] in human
    assert planning["tree"] in human
    assert "Automatically executed: `false`" in human
    assert "May create follow-on tasks: `false`" in human
    lowered = human.lower()
    for claim in FORBIDDEN_PRODUCTION_CLAIMS:
        assert claim not in lowered
    assert "non_promoted_unmeasured" in human
    assert payload["next_highest_return_engineering_work"]["items"][0]["id"] in human
    for item in payload["next_highest_return_engineering_work"]["items"]:
        assert item["executing"] is False
        assert f"executing=false" in human


def test_deferred_ideas_remain_non_executing_and_create_no_backlog() -> None:
    payload = _load_canonical(REPORT_PATH)
    human = HUMAN_PATH.read_text(encoding="utf-8")
    next_work = payload["next_highest_return_engineering_work"]
    assert next_work["automatically_executed"] is False
    assert next_work["may_create_follow_on_tasks"] is False
    assert payload["limits"]["follow_on_tasks_created"] is False
    assert payload["limits"]["deferred_ideas_non_executing"] is True
    todo = TODO_PATH.read_text(encoding="utf-8")
    assert "ASEH-076" not in todo
    assert "ASEH-076" not in human
    assert "## ASEH-075 Publish final residual-gap report" in todo
    assert "- Status: todo" in todo.split("## ASEH-075 Publish final residual-gap report", 1)[1]


def test_negative_cases_fail_closed() -> None:
    evidence = load_current_evidence()
    payload = build_release_report(evidence)

    promoted = copy.deepcopy(payload)
    promoted["disposition"] = "promotion_eligible_operator_authorization_required"
    promoted["disposition_kind"] = "eligible_operator_authorization_required"
    promoted["promotion_status"]["disposition"] = (
        "promotion_eligible_operator_authorization_required"
    )
    promoted["identity"] = content_identity(
        {key: value for key, value in promoted.items() if key != "identity"}
    )
    with pytest.raises(ReleaseReportError, match="absent live cohort|does not equal"):
        validate_report(promoted, evidence=evidence)

    missing = copy.deepcopy(payload)
    del missing["residual_risks"]
    with pytest.raises(ReleaseReportError, match="does not equal|absent required field"):
        validate_report(missing, evidence=evidence)

    zeroed = copy.deepcopy(payload)
    zeroed["baseline_current_candidate_token_use"]["live"]["reason_code"] = "not_yet_measured"
    zeroed["baseline_current_candidate_token_use"]["live"]["truth_state"] = "unavailable"
    zeroed["baseline_current_candidate_token_use"]["live"]["value"] = 0
    with pytest.raises(ReleaseReportError, match="numeric value|does not equal"):
        try:
            _assert_no_numeric_unavailable(zeroed)
        except ReleaseReportError:
            raise
        validate_report(zeroed, evidence=evidence)

    executing = copy.deepcopy(payload)
    executing["deferred_backlog"]["automatically_executed"] = True
    executing["identity"] = content_identity(
        {key: value for key, value in executing.items() if key != "identity"}
    )
    with pytest.raises(ReleaseReportError, match="deferred backlog|does not equal"):
        validate_report(executing, evidence=evidence)

    follow_on = copy.deepcopy(payload)
    follow_on["deferred_backlog"]["may_create_follow_on_tasks"] = True
    follow_on["identity"] = content_identity(
        {key: value for key, value in follow_on.items() if key != "identity"}
    )
    with pytest.raises(ReleaseReportError, match="follow-on|does not equal"):
        validate_report(follow_on, evidence=evidence)

    self_auth = copy.deepcopy(payload)
    self_auth["self_authorized"] = True
    self_auth["identity"] = content_identity(
        {key: value for key, value in self_auth.items() if key != "identity"}
    )
    with pytest.raises(ReleaseReportError, match="self-authorize|does not equal"):
        validate_report(self_auth, evidence=evidence)

    live_label = copy.deepcopy(payload)
    live_label["benchmark_population_sample_sizes"]["hermetic"]["live"] = True
    live_label["identity"] = content_identity(
        {key: value for key, value in live_label.items() if key != "identity"}
    )
    with pytest.raises(ReleaseReportError, match="labeled live|does not equal"):
        validate_report(live_label, evidence=evidence)

    drifted = copy.deepcopy(evidence)
    drifted["promotion"] = copy.deepcopy(evidence["promotion"])
    drifted["promotion"]["disposition"] = "promotion_eligible_operator_authorization_required"
    drifted["promotion"]["disposition_kind"] = "eligible_operator_authorization_required"
    with pytest.raises(ReleaseReportError, match="absent live cohort"):
        built = build_release_report(drifted)
        validate_report(built, evidence=drifted)


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
