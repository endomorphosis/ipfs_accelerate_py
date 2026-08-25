"""Closed, fail-closed contract checks for the CASF-043 qualification report."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = ROOT / "docs/architecture/causal_event_federation_inventory/final_qualification_report.json"
MARKDOWN_PATH = ROOT / "docs/architecture/causal_event_federation_inventory/final_qualification_report.md"
BENCHMARK_SUITE_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/manifest.json"

STARTING_REVISION = "84a056e41e48a81d4484be43840196578d6c87da"
STARTING_TREE = "40f0771e77d394ac91d92cc1edb02f7860f6131b"
OBSERVED_REVISION = "5796d3f78b77b2b6c1c59a2b74c86020a0b141ae"
OBSERVED_TREE = "14b36ca1f21bfd03dd4b88a7866a0c1a40059249"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OID_RE = re.compile(r"^[0-9a-f]{40}$")

TOP_LEVEL_FIELDS = frozenset(
    {
        "artifact_type",
        "schema",
        "artifact_version",
        "program_id",
        "root_objective_id",
        "task_id",
        "authority",
        "baseline",
        "observed_component_snapshot",
        "control_plane",
        "capabilities",
        "population_and_concurrency",
        "benchmark_evidence",
        "result_coverage",
        "execution_evidence",
        "safety_gates",
        "claims",
        "residual_gaps",
        "qualification",
        "rollback",
        "nonclaims",
    }
)
CLAIM_FIELDS = frozenset(
    {
        "event_driven",
        "causally_coordinated",
        "multi_supervisor",
        "parallel",
        "token_efficient",
        "production_ready",
        "ducklake_promotion_qualified",
        "exactly_once_network_delivery",
    }
)
RESULT_FIELDS = frozenset(
    {
        "task_and_dedup",
        "causal_graph_and_abstraction",
        "interventions_and_independence",
        "events_outbox_dead_letters_and_wakeups",
        "idle_behavior",
        "parallel_throughput_and_merge",
        "model_context_and_token_efficiency",
        "proof_and_validation",
        "ducklake_projection",
        "recovery_and_failures",
    }
)
SAFETY_GATES = frozenset(
    {
        "direct_multi_process_file_mutation",
        "store_ambiguity",
        "event_loss",
        "duplicate_committed_effects",
        "stale_fence_completion",
        "unauthorized_creation",
        "tenant_leakage",
        "agent_sql",
        "secret_leakage",
        "causal_notification_loss",
        "nomination_or_stale_map_authority",
        "cycle_or_shard_corruption",
        "forbidden_idle_activity",
        "replay_idempotency",
        "ownership_effect_or_merge_corruption",
        "reduced_assurance",
    }
)
GAP_IDS = frozenset(
    {
        "CASF-043-EXACT-MERGED-TREE-IDENTITY",
        "CASF-043-LIVE-TYPED-QUACK",
        "CASF-043-SCALE-BENCHMARKS-NOT-RUN",
        "CASF-043-TOKEN-BENCHMARK-NOT-RUN",
        "CASF-043-CONJUNCTIVE-GATE-DECISION",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _require_exact_keys(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has unknown, missing, or non-object fields")
    return value


def _validate_report(report: object) -> dict[str, Any]:
    value = _require_exact_keys(report, TOP_LEVEL_FIELDS, "qualification report")
    if value["artifact_type"] != "casf_final_qualification_report":
        raise ValueError("wrong qualification report artifact type")
    if value["schema"] != "casf/qualification-report@1" or value["artifact_version"] != 1:
        raise ValueError("wrong qualification report schema")
    if value["program_id"] != "agent-supervisor-causal-event-federation-v1":
        raise ValueError("wrong program identity")
    if value["root_objective_id"] != "CASF-G000" or value["task_id"] != "CASF-043":
        raise ValueError("wrong task identity")

    authority = _require_exact_keys(
        value["authority"],
        frozenset(
            {
                "authoritative",
                "qualification_authority",
                "completion_authority",
                "promotion_authority",
                "scheduling_authority",
                "reason",
            }
        ),
        "authority",
    )
    if any(authority[name] is not False for name in authority if name != "reason"):
        raise ValueError("report must not create authority")

    baseline = _require_exact_keys(
        value["baseline"], frozenset({"revision", "tree_id", "source"}), "baseline"
    )
    if baseline != {
        "revision": STARTING_REVISION,
        "tree_id": STARTING_TREE,
        "source": "docs/architecture/causal_event_federation_inventory/starting_tree.json",
    }:
        raise ValueError("starting baseline is not sealed")
    observed = _require_exact_keys(
        value["observed_component_snapshot"],
        frozenset(
            {
                "revision",
                "tree_id",
                "observation_kind",
                "observed_at_utc",
                "report_artifact_in_snapshot",
                "qualification_identity_available",
                "reason",
            }
        ),
        "observed component snapshot",
    )
    if observed["revision"] != OBSERVED_REVISION or observed["tree_id"] != OBSERVED_TREE:
        raise ValueError("observed component snapshot is not exact")
    if not OID_RE.fullmatch(observed["revision"]) or not OID_RE.fullmatch(observed["tree_id"]):
        raise ValueError("snapshot identities must be Git object IDs")
    if (
        observed["observation_kind"] != "committed_source_tree_before_report_artifact"
        or observed["report_artifact_in_snapshot"] is not False
        or observed["qualification_identity_available"] is not False
    ):
        raise ValueError("report must not self-promote its snapshot")

    control = _require_exact_keys(
        value["control_plane"],
        frozenset({"state", "generation_id", "schema_fingerprint", "reason"}),
        "control-plane observation",
    )
    if (
        control["state"] != "not_observed"
        or control["generation_id"] is not None
        or control["schema_fingerprint"] is not None
    ):
        raise ValueError("unattested control-plane identity must remain unobserved")

    capabilities = _require_exact_keys(
        value["capabilities"],
        frozenset({"state_owner", "quack_event_wait", "ducklake_projection", "live_scale_capacity"}),
        "capabilities",
    )
    expected_capabilities = {
        "state_owner": "not_live_qualified",
        "quack_event_wait": "owner_local_hermetic_only",
        "ducklake_projection": "not_currently_qualified",
        "live_scale_capacity": "unavailable",
    }
    for name, status in expected_capabilities.items():
        capability = _require_exact_keys(
            capabilities[name], frozenset({"status", "evidence_ref", "reason"}), name
        )
        if capability["status"] != status or capability["evidence_ref"] is not None:
            raise ValueError(f"{name} capability was overstated")

    population = _require_exact_keys(
        value["population_and_concurrency"],
        frozenset(
            {
                "federation_population",
                "supervisor_processes",
                "registered_logical_agents",
                "maximum_concurrent_subagents",
                "qualified_live_population",
                "reason",
            }
        ),
        "population and concurrency",
    )
    if (
        any(population[name] != "not_observed" for name in (
            "federation_population",
            "supervisor_processes",
            "registered_logical_agents",
            "maximum_concurrent_subagents",
        ))
        or population["qualified_live_population"] is not False
    ):
        raise ValueError("unattested live population must not be reported")

    benchmarks = _require_exact_keys(
        value["benchmark_evidence"],
        frozenset(
            {
                "suite_path",
                "suite_component_snapshot",
                "suite_current_for_observed_component_snapshot",
                "suite_status",
                "authoritative",
                "promotion_eligible",
                "components",
            }
        ),
        "benchmark evidence",
    )
    if (
        benchmarks["suite_path"] != "benchmarks/agent_supervisor/causal_event_federation/manifest.json"
        or benchmarks["suite_current_for_observed_component_snapshot"] is not False
        or benchmarks["suite_status"] != "not_run"
        or benchmarks["authoritative"] is not False
        or benchmarks["promotion_eligible"] is not False
        or type(benchmarks["components"]) is not list
    ):
        raise ValueError("benchmark suite was overstated")
    suite_snapshot = _require_exact_keys(
        benchmarks["suite_component_snapshot"], frozenset({"revision", "tree_id"}), "suite snapshot"
    )
    if suite_snapshot == {"revision": OBSERVED_REVISION, "tree_id": OBSERVED_TREE}:
        raise ValueError("stale benchmark suite cannot become current evidence")
    if len(benchmarks["components"]) != 4:
        raise ValueError("benchmark evidence must cover every CASF benchmark")
    expected_benchmarks = {
        "CASF-038": ("casf/idle-benchmark@1", "typed_quack_live_endpoint_not_supplied"),
        "CASF-039": ("casf/parallel-benchmark@1", "qualified_twelve_supervisor_live_capacity_not_admitted"),
        "CASF-040": ("casf/load-benchmark@1", "qualified_256_agent_bounded_load_live_capacity_not_admitted"),
        "CASF-041": ("casf/token-benchmark@1", "qualified_cross_supervisor_token_live_capacity_not_admitted"),
    }
    for component in benchmarks["components"]:
        component = _require_exact_keys(
            component,
            frozenset(
                {"task_id", "schema", "availability", "execution_status", "qualified", "promotion_eligible", "reason_code"}
            ),
            "benchmark component",
        )
        if component["task_id"] not in expected_benchmarks:
            raise ValueError("unknown benchmark component")
        schema, reason = expected_benchmarks[component["task_id"]]
        if (
            component["schema"] != schema
            or component["reason_code"] != reason
            or component["availability"] != "unavailable"
            or component["execution_status"] != "not_run"
            or component["qualified"] is not False
            or component["promotion_eligible"] is not False
        ):
            raise ValueError("benchmark component was overstated")

    if set(value["result_coverage"]) != RESULT_FIELDS or any(
        item != "not_currently_qualified" for item in value["result_coverage"].values()
    ):
        raise ValueError("result coverage must fail closed")
    if set(value["safety_gates"]) != SAFETY_GATES or any(
        item != "unverified" for item in value["safety_gates"].values()
    ):
        raise ValueError("safety gates must not be inferred")
    if set(value["claims"]) != CLAIM_FIELDS or any(item is not False for item in value["claims"].values()):
        raise ValueError("unsupported readiness claim")

    execution = _require_exact_keys(
        value["execution_evidence"],
        frozenset({"executed_current_tree_receipts", "unexecuted_or_not_current", "models_executed", "reason"}),
        "execution evidence",
    )
    if execution["executed_current_tree_receipts"] != [] or execution["models_executed"] != []:
        raise ValueError("report must not invent execution evidence")
    if set(execution["unexecuted_or_not_current"]) != {
        "casf/idle-benchmark@1",
        "casf/parallel-benchmark@1",
        "casf/load-benchmark@1",
        "casf/token-benchmark@1",
        "casf/promotion-decision@1",
        "casf/federation-completion-receipt@1",
    }:
        raise ValueError("unexecuted evidence coverage is incomplete")

    gaps = value["residual_gaps"]
    if type(gaps) is not list or len(gaps) != len(GAP_IDS):
        raise ValueError("residual gaps must be complete")
    if {item.get("gap_id") for item in gaps if type(item) is dict} != GAP_IDS:
        raise ValueError("residual gaps are incomplete or unknown")
    for gap in gaps:
        gap = _require_exact_keys(
            gap, frozenset({"gap_id", "status", "scope", "required_resolution"}), "residual gap"
        )
        if gap["status"] != "blocking" or not gap["scope"] or not gap["required_resolution"]:
            raise ValueError("residual gaps must be actionable blockers")

    qualification = _require_exact_keys(
        value["qualification"],
        frozenset(
            {
                "status",
                "promotion_eligible",
                "release_eligible",
                "quarantine_required",
                "promotion_applied",
                "rollback_applied",
                "authoritative_state_changed",
                "completion_created",
                "upstream_reverification_required",
                "reason",
            }
        ),
        "qualification disposition",
    )
    if qualification != {
        **qualification,
        "status": "blocked",
        "promotion_eligible": False,
        "release_eligible": False,
        "quarantine_required": True,
        "promotion_applied": False,
        "rollback_applied": False,
        "authoritative_state_changed": False,
        "completion_created": False,
        "upstream_reverification_required": True,
    }:
        raise ValueError("qualification disposition was overstated")

    rollback = _require_exact_keys(
        value["rollback"],
        frozenset({"status", "target", "required_authority", "history_rewrite_permitted"}),
        "rollback boundary",
    )
    if rollback["status"] != "not_authorized_by_this_report" or rollback["history_rewrite_permitted"] is not False:
        raise ValueError("report must not authorize rollback")
    if rollback["target"] != {"revision": STARTING_REVISION, "tree_id": STARTING_TREE}:
        raise ValueError("rollback target must be the sealed baseline")
    if type(value["nonclaims"]) is not list or len(value["nonclaims"]) < 4:
        raise ValueError("non-authority constraints are incomplete")
    return value


def test_report_is_closed_source_bound_and_fail_closed() -> None:
    report = _validate_report(_read_json(REPORT_PATH))
    suite = _read_json(BENCHMARK_SUITE_PATH)
    assert suite["component_snapshot"] == report["benchmark_evidence"]["suite_component_snapshot"]
    assert suite["suite_status"] == report["benchmark_evidence"]["suite_status"]
    assert suite["promotion_eligible"] is report["benchmark_evidence"]["promotion_eligible"]


def test_markdown_is_bound_to_machine_report_and_states_blocked_disposition() -> None:
    machine_bytes = REPORT_PATH.read_bytes()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    digest = hashlib.sha256(machine_bytes).hexdigest()

    assert SHA256_RE.fullmatch(digest)
    assert f"Machine-report SHA-256: `{digest}`" in markdown
    for required_text in (
        "Blocked — quarantine required.",
        OBSERVED_REVISION,
        OBSERVED_TREE,
        "not a remote multi-supervisor qualification",
        "CASF-043-EXACT-MERGED-TREE-IDENTITY",
        "history rewriting is prohibited",
    ):
        assert required_text in markdown


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["claims"].update({"production_ready": True}),
        lambda report: report["authority"].update({"completion_authority": True}),
        lambda report: report["observed_component_snapshot"].update({"qualification_identity_available": True}),
        lambda report: report["capabilities"]["quack_event_wait"].update({"status": "live_qualified"}),
        lambda report: report["benchmark_evidence"]["components"][1].update({"qualified": True}),
        lambda report: report["safety_gates"].update({"event_loss": "passed"}),
        lambda report: report["residual_gaps"].pop(),
        lambda report: report["qualification"].update({"promotion_eligible": True}),
        lambda report: report["rollback"].update({"history_rewrite_permitted": True}),
    ],
)
def test_report_rejects_unsupported_claims_and_missing_gaps(mutate: Any) -> None:
    report = deepcopy(_read_json(REPORT_PATH))
    mutate(report)

    with pytest.raises(ValueError):
        _validate_report(report)


def test_report_rejects_unknown_normative_fields() -> None:
    report = deepcopy(_read_json(REPORT_PATH))
    report["model_declared_ready"] = True

    with pytest.raises(ValueError, match="unknown"):
        _validate_report(report)
