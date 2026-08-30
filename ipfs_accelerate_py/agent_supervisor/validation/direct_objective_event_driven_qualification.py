"""Fail-closed PCPR Phase-0 supervisor qualification.

Qualify the existing direct-objective submission path and event-driven
planning/refill machinery against the required cohort and efficiency
targets.  This module is not release authority: it does not write DuckDB
or Quack state, does not freeze contracts, and never emits a closed PCPR
release outcome.

Live claims require ``measured_live`` evidence.  Hermetic tests are
recorded as ``measured_hermetic`` and cannot satisfy live cohort counts
or efficiency targets.  Simulated, estimated, and unavailable values are
never represented as zero, passing, or live.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


DIRECT_OBJECTIVE_EVENT_DRIVEN_QUALIFICATION_INTERFACE: Final = (
    "DirectObjectiveEventDrivenQualification@1"
)
DIRECT_OBJECTIVE_EVENT_DRIVEN_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "direct-objective-event-driven-qualification@1"
)
QUALIFICATION_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "direct-objective-event-driven-qualification-verdict@1"
)

EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "measured_live",
        "measured_hermetic",
        "estimated",
        "simulated",
        "unavailable",
    }
)
LIVE_SATISFYING_KIND: Final = "measured_live"

PROMOTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "supervisor_promoted",
        "supervisor_non_promoted",
        "rnd_non_promoted",
        "typed_unavailable",
        "typed_blocked",
    }
)

# Closed campaign outcomes from the PCPR scheduler.  A Phase-0 R&D
# qualification receipt must not mint any of these.
CLOSED_RELEASE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "release_candidate_qualified",
        "non_promoted_supervisor_unqualified",
        "non_promoted_import_or_false_success",
        "non_promoted_live_storage_gap",
        "non_promoted_live_compute_gap",
        "non_promoted_solver_gap",
        "non_promoted_packaging_gap",
        "non_promoted_dependency_reproducibility",
        "non_promoted_security_failure",
        "non_promoted_interoperability_gap",
        "non_promoted_reference_workflow_failure",
        "non_promoted_unmeasured",
        "non_promoted_operator_gate_required",
    }
)

REQUIRED_COHORT_CASES: Final[tuple[str, ...]] = (
    "ten_consecutive_bounded_objectives",
    "twenty_historical_task_replays",
    "held_out_high_level_decomposition",
    "owner_loss_and_restart",
    "stale_task_recovery",
    "provider_outcome_unknown_reconciliation",
    "lease_and_fencing_races",
    "event_replay_and_duplicate_delivery",
    "automatic_task_frontier_refill",
    "incremental_plan_reassessment",
    "contextpack_reuse_and_invalidation",
    "cross_supervisor_event_handling",
    "external_python_and_mcp_submission",
)

# Live cohort minima.  Hermetic unit coverage cannot substitute.
LIVE_OBJECTIVE_MINIMUM: Final[int] = 10
LIVE_REPLAY_MINIMUM: Final[int] = 20

# Efficiency targets in integer basis points (1 percent = 100 bps) except
# the two hard-zero counters and net-cost, which use integer units.
TARGET_MEDIAN_INPUT_TOKEN_REDUCTION_BPS: Final[int] = 3_000
TARGET_FRONTIER_MODEL_CALL_REDUCTION_BPS: Final[int] = 4_000
TARGET_ELIGIBLE_DECISIONS_WITHOUT_FRONTIER_BPS: Final[int] = 6_000
TARGET_ORDINARY_REFILLS_WITHOUT_LLM_BPS: Final[int] = 8_000
TARGET_CONTEXTPACK_REUSE_BPS: Final[int] = 5_000
TARGET_MAX_UNNECESSARY_TASK_CHURN_BPS: Final[int] = 500
TARGET_MAX_MANUAL_RECOVERY_BPS: Final[int] = 200
TARGET_MANUAL_TASK_TABLE_EDITS: Final[int] = 0
TARGET_HARD_SAFETY_FAILURES: Final[int] = 0

REQUIRED_TARGETS: Final[tuple[str, ...]] = (
    "median_end_to_end_input_token_reduction",
    "frontier_model_call_reduction",
    "net_cost_reduction_after_audit",
    "eligible_decisions_without_frontier_model",
    "ordinary_refills_without_llm",
    "contextpack_reuse_on_eligible_tasks",
    "unnecessary_task_churn",
    "manual_recovery",
    "manual_task_table_edits",
    "hard_safety_failures",
)

# Outer-receipt narrative for the ordinary missing-live-campaign case.
# These strings are evidence labels, not measurements.
CASE_UNAVAILABLE_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ten_consecutive_bounded_objectives": (
            "No live sequence of 10 consecutive bounded objectives ran in this "
            "environment. Hermetic single-objective prompt tests are not substituted."
        ),
        "twenty_historical_task_replays": (
            "No live historical 20-task replay cohort ran. Event-log cursor replay "
            "tests are hermetic candidates only."
        ),
        "held_out_high_level_decomposition": (
            "No held-out live high-level objective decomposition campaign ran."
        ),
        "owner_loss_and_restart": (
            "No live owner-loss or authoritative-state-owner restart campaign ran."
        ),
        "stale_task_recovery": (
            "No live stale-task recovery campaign ran. Gitlink landed-merge "
            "reconciliation tests are hermetic candidates only."
        ),
        "provider_outcome_unknown_reconciliation": (
            "No live provider-outcome-unknown reconciliation campaign ran."
        ),
        "lease_and_fencing_races": "No live lease/fencing race campaign ran.",
        "event_replay_and_duplicate_delivery": (
            "No live duplicate-delivery campaign ran. Hermetic cursor replay is not live."
        ),
        "automatic_task_frontier_refill": (
            "No live automatic task-frontier refill campaign ran. Post-merge auto-start, "
            "portal-idle, and gitlink landed-merge tests are hermetic candidates only."
        ),
        "incremental_plan_reassessment": (
            "No live incremental plan-reassessment campaign ran."
        ),
        "contextpack_reuse_and_invalidation": (
            "No live ContextPack reuse/invalidation campaign ran."
        ),
        "cross_supervisor_event_handling": (
            "No live cross-supervisor event campaign ran."
        ),
        "external_python_and_mcp_submission": (
            "Hermetic Python/CLI/MCP identity tests exist and were exercised; they "
            "are not a live external-client campaign."
        ),
    }
)

TARGET_UNAVAILABLE_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "median_end_to_end_input_token_reduction": (
            "No live Codex-primed baseline or current-route token measurements exist. "
            "Missing is not recorded as 0%."
        ),
        "frontier_model_call_reduction": (
            "No live frontier-model call ledger was observed. Missing is not recorded as 0%."
        ),
        "net_cost_reduction_after_audit": (
            "No live provider-cost ledger was observed. Missing is not recorded as zero savings."
        ),
        "eligible_decisions_without_frontier_model": (
            "No live eligible-decision ledger was observed. Missing is not recorded as 0%."
        ),
        "ordinary_refills_without_llm": (
            "No live ordinary-refill ledger was observed. Missing is not recorded as 0%."
        ),
        "contextpack_reuse_on_eligible_tasks": (
            "No live ContextPack reuse ledger was observed. Missing is not recorded as 0%."
        ),
        "unnecessary_task_churn": (
            "No live unnecessary-task-churn ledger was observed. Missing is not recorded as 0%."
        ),
        "manual_recovery": (
            "No live manual-recovery ledger was observed. Missing is not recorded as 0%."
        ),
        "manual_task_table_edits": (
            "This worker did not write DuckDB or Quack state. That local fact is not "
            "substituted for a live 10-objective campaign counter."
        ),
        "hard_safety_failures": (
            "Live campaign hard-zero counters were not observed and are not recorded as zero."
        ),
    }
)

HARD_ZERO_INVARIANTS: Final[tuple[str, ...]] = (
    "false_completions",
    "unauthorized_mutations",
    "simulated_as_live",
    "stale_cache_admissions",
    "stale_contextpack_admissions",
    "stale_lease_completions",
    "stale_fence_completions",
    "double_execution",
    "double_terminalization",
    "confirmation_replays",
    "path_or_scope_escapes",
    "hidden_validation_reductions",
    "accepted_critical_controlled_omissions",
    "selected_test_false_negatives_in_release_qualification",
    "self_authorized_promotions",
    "release_creation_after_failed_required_gates",
)

# Post-merge gitlink landed-completion coverage.  Trailing-slash gitlink
# outputs and completed-merge proofs are hermetic candidates only.
GITLINK_LANDED_MERGE_HERMETIC_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
)

# Existing hermetic suites that exercise machinery related to each case.
# Presence of a path is not live qualification and is not a pass.
HERMETIC_CANDIDATE_SUITES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "ten_consecutive_bounded_objectives": (
            "test/api/test_agent_supervisor_prompt_v3_python_api.py",
            "test/api/test_agent_supervisor_prompt_v3_cli.py",
            "test/api/test_agent_supervisor_prompt_v3_mcp.py",
        ),
        "twenty_historical_task_replays": (
            "test/api/test_agent_supervisor_event_driven_runtime.py",
        ),
        "held_out_high_level_decomposition": (
            "test/api/test_agent_supervisor_prompt_v3_python_api.py",
            "test/integration/test_agent_supervisor_planner_doctor_e2e.py",
        ),
        "owner_loss_and_restart": (
            "test/api/test_agent_supervisor_fault_recovery_v2.py",
            "test/api/test_agent_supervisor_daemon_restart_durability.py",
        ),
        "stale_task_recovery": (
            "test/api/test_agent_supervisor_acceptance_recovery.py",
            "test/api/test_agent_supervisor_typed_deferral_recovery.py",
            *GITLINK_LANDED_MERGE_HERMETIC_SUITES,
        ),
        "provider_outcome_unknown_reconciliation": (
            "test/api/test_agent_supervisor_typed_deferral_recovery.py",
            "test/api/test_agent_supervisor_autonomous_unstall.py",
        ),
        "lease_and_fencing_races": (
            "test/api/test_agent_supervisor_lease_coordination.py",
            "test/api/test_agent_supervisor_distributed_lanes.py",
        ),
        "event_replay_and_duplicate_delivery": (
            "test/api/test_agent_supervisor_event_driven_runtime.py",
        ),
        "automatic_task_frontier_refill": (
            "test/api/test_agent_supervisor_plan_supervisor_service.py",
            "test/api/test_agent_supervisor_post_completion_ops_refill.py",
            "test/api/test_agent_supervisor_refill_residual_guard.py",
            "test/api/test_agent_supervisor_todo_daemon_port.py",
            "test/api/test_agent_supervisor_database_portal_bridge.py",
            *GITLINK_LANDED_MERGE_HERMETIC_SUITES,
        ),
        "incremental_plan_reassessment": (
            "test/api/test_agent_supervisor_adaptive_planner.py",
            "test/api/test_agent_supervisor_plan_supervisor_service.py",
        ),
        "contextpack_reuse_and_invalidation": (
            "test/api/test_agent_supervisor_context_delta.py",
            "test/api/test_agent_supervisor_planner_doctor_context.py",
        ),
        "cross_supervisor_event_handling": (
            "test/api/test_agent_supervisor_multi_runner_bootstrap_fd.py",
            "test/api/test_agent_supervisor_implementation_supervisor_authority_forwarding.py",
        ),
        "external_python_and_mcp_submission": (
            "test/api/test_agent_supervisor_prompt_v3_python_api.py",
            "test/api/test_agent_supervisor_prompt_v3_cli.py",
            "test/api/test_agent_supervisor_prompt_v3_mcp.py",
            "test/api/test_agent_supervisor_plan_control_conformance.py",
        ),
    }
)


class DirectObjectiveEventDrivenQualificationError(ValueError):
    """Malformed qualification evidence or measurement."""


@dataclass(frozen=True)
class CohortEvidence:
    """One required cohort case with an explicit evidence kind."""

    case_id: str
    evidence_kind: str
    status: str
    reason: str
    live_count: int | None = None
    live_environment_id: str = ""
    hermetic_suite_paths: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "evidence_kind": self.evidence_kind,
            "status": self.status,
            "reason": self.reason,
            "live_environment_id": self.live_environment_id,
            "hermetic_suite_paths": list(self.hermetic_suite_paths),
        }
        if self.live_count is None:
            payload["live_count"] = None
        else:
            payload["live_count"] = int(self.live_count)
        return payload


@dataclass(frozen=True)
class TargetMeasurement:
    """One efficiency or hard-zero target with an explicit evidence kind."""

    target_id: str
    evidence_kind: str
    observed_bps: int | None = None
    observed_count: int | None = None
    observed_net_cost_units: int | None = None
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "evidence_kind": self.evidence_kind,
            "observed_bps": self.observed_bps,
            "observed_count": self.observed_count,
            "observed_net_cost_units": self.observed_net_cost_units,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SafetyVector:
    """Hard-zero invariant counters.  Missing entries are unavailable."""

    counts: Mapping[str, int]
    evidence_kind: str
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "counts": {name: int(self.counts[name]) for name in HARD_ZERO_INVARIANTS if name in self.counts},
            "missing": [name for name in HARD_ZERO_INVARIANTS if name not in self.counts],
            "evidence_kind": self.evidence_kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QualificationVerdict:
    """Fail-closed Phase-0 qualification decision."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    cohort: tuple[CohortEvidence, ...]
    targets: tuple[TargetMeasurement, ...]
    safety: SafetyVector
    missed_live_cohort: tuple[str, ...]
    missed_targets: tuple[str, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    live_campaign_identity: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "cohort": [item.to_mapping() for item in self.cohort],
            "targets": [item.to_mapping() for item in self.targets],
            "safety": self.safety.to_mapping(),
            "missed_live_cohort": list(self.missed_live_cohort),
            "missed_targets": list(self.missed_targets),
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
            "live_campaign_identity": self.live_campaign_identity,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DirectObjectiveEventDrivenQualificationError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise DirectObjectiveEventDrivenQualificationError(f"{name} is not an admitted evidence kind")
    return kind


def _status(value: Any) -> str:
    status = _text(value, "status")
    if status not in {"passed", "failed", "unavailable", "blocked"}:
        raise DirectObjectiveEventDrivenQualificationError("status is not an admitted cohort status")
    return status


def _optional_int(value: Any, name: str, *, non_negative: bool) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DirectObjectiveEventDrivenQualificationError(f"{name} must be an integer or null")
    if non_negative and value < 0:
        raise DirectObjectiveEventDrivenQualificationError(f"{name} must be non-negative")
    return value


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    return _optional_int(value, name, non_negative=True)


def _require_unique(ids: Sequence[str], population: Sequence[str], name: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in ids:
        if item in seen:
            raise DirectObjectiveEventDrivenQualificationError(f"duplicate {name}: {item}")
        seen.add(item)
        ordered.append(item)
    missing = [item for item in population if item not in seen]
    extra = [item for item in ordered if item not in population]
    if extra:
        raise DirectObjectiveEventDrivenQualificationError(f"unknown {name}: {extra[0]}")
    if missing:
        raise DirectObjectiveEventDrivenQualificationError(f"missing {name}: {missing[0]}")
    return tuple(ordered)


def unavailable_cohort_evidence(
    case_id: str,
    *,
    reason: str,
) -> CohortEvidence:
    """Typed unavailable live evidence for one required cohort case."""

    if case_id not in REQUIRED_COHORT_CASES:
        raise DirectObjectiveEventDrivenQualificationError(f"unknown cohort case: {case_id}")
    return CohortEvidence(
        case_id=case_id,
        evidence_kind="unavailable",
        status="unavailable",
        reason=reason,
        live_count=None,
        live_environment_id="",
        hermetic_suite_paths=HERMETIC_CANDIDATE_SUITES[case_id],
    )


def unavailable_target_measurement(target_id: str, *, reason: str) -> TargetMeasurement:
    """Typed unavailable live measurement for one required target."""

    if target_id not in REQUIRED_TARGETS:
        raise DirectObjectiveEventDrivenQualificationError(f"unknown target: {target_id}")
    return TargetMeasurement(
        target_id=target_id,
        evidence_kind="unavailable",
        reason=reason,
    )


def unavailable_safety_vector(*, reason: str) -> SafetyVector:
    """Hard-zero vector whose live campaign counters were not observed."""

    return SafetyVector(counts=MappingProxyType({}), evidence_kind="unavailable", reason=reason)


def current_head_unavailable_inputs(
    *,
    reason: str = "",
) -> tuple[tuple[CohortEvidence, ...], tuple[TargetMeasurement, ...], SafetyVector]:
    """Default fail-closed inputs when the live campaign is absent."""

    safety_reason = reason or (
        "No live PCPR Phase-0 cohort ran in this environment; hermetic suites "
        "remain candidate coverage and are not substituted for live counts or "
        "efficiency targets."
    )
    cohort = tuple(
        unavailable_cohort_evidence(
            case_id,
            reason=reason or CASE_UNAVAILABLE_REASONS[case_id],
        )
        for case_id in REQUIRED_COHORT_CASES
    )
    targets = tuple(
        unavailable_target_measurement(
            target_id,
            reason=reason or TARGET_UNAVAILABLE_REASONS[target_id],
        )
        for target_id in REQUIRED_TARGETS
    )
    return cohort, targets, unavailable_safety_vector(reason=safety_reason)


def _live_cohort_satisfied(record: CohortEvidence) -> bool:
    if record.evidence_kind != LIVE_SATISFYING_KIND or record.status != "passed":
        return False
    if not record.live_environment_id:
        return False
    if record.case_id == "ten_consecutive_bounded_objectives":
        return record.live_count is not None and record.live_count >= LIVE_OBJECTIVE_MINIMUM
    if record.case_id == "twenty_historical_task_replays":
        return record.live_count is not None and record.live_count >= LIVE_REPLAY_MINIMUM
    return True


def _target_satisfied(record: TargetMeasurement) -> bool:
    if record.evidence_kind != LIVE_SATISFYING_KIND:
        return False
    target_id = record.target_id
    if target_id == "median_end_to_end_input_token_reduction":
        return (
            record.observed_bps is not None
            and record.observed_bps >= TARGET_MEDIAN_INPUT_TOKEN_REDUCTION_BPS
        )
    if target_id == "frontier_model_call_reduction":
        return (
            record.observed_bps is not None
            and record.observed_bps >= TARGET_FRONTIER_MODEL_CALL_REDUCTION_BPS
        )
    if target_id == "net_cost_reduction_after_audit":
        return (
            record.observed_net_cost_units is not None
            and record.observed_net_cost_units > 0
        )
    if target_id == "eligible_decisions_without_frontier_model":
        return (
            record.observed_bps is not None
            and record.observed_bps >= TARGET_ELIGIBLE_DECISIONS_WITHOUT_FRONTIER_BPS
        )
    if target_id == "ordinary_refills_without_llm":
        return (
            record.observed_bps is not None
            and record.observed_bps >= TARGET_ORDINARY_REFILLS_WITHOUT_LLM_BPS
        )
    if target_id == "contextpack_reuse_on_eligible_tasks":
        return (
            record.observed_bps is not None
            and record.observed_bps >= TARGET_CONTEXTPACK_REUSE_BPS
        )
    if target_id == "unnecessary_task_churn":
        return (
            record.observed_bps is not None
            and record.observed_bps <= TARGET_MAX_UNNECESSARY_TASK_CHURN_BPS
        )
    if target_id == "manual_recovery":
        return (
            record.observed_bps is not None
            and record.observed_bps <= TARGET_MAX_MANUAL_RECOVERY_BPS
        )
    if target_id == "manual_task_table_edits":
        return (
            record.observed_count is not None
            and record.observed_count == TARGET_MANUAL_TASK_TABLE_EDITS
        )
    if target_id == "hard_safety_failures":
        return (
            record.observed_count is not None
            and record.observed_count == TARGET_HARD_SAFETY_FAILURES
        )
    return False


def _normalize_cohort(records: Sequence[CohortEvidence]) -> tuple[CohortEvidence, ...]:
    normalized: list[CohortEvidence] = []
    for record in records:
        case_id = _text(record.case_id, "case_id")
        kind = _kind(record.evidence_kind, "evidence_kind")
        status = _status(record.status)
        reason = _text(record.reason, "reason")
        live_count = _optional_non_negative_int(record.live_count, "live_count")
        environment = record.live_environment_id
        if environment is None or not isinstance(environment, str):
            raise DirectObjectiveEventDrivenQualificationError(
                "live_environment_id must be a string"
            )
        if kind == LIVE_SATISFYING_KIND and not environment.strip():
            raise DirectObjectiveEventDrivenQualificationError(
                "measured_live evidence requires live_environment_id"
            )
        if kind == "simulated" and status == "passed":
            # Simulated success is retained as simulated; it cannot pass live.
            status = "unavailable"
            reason = "simulated results cannot satisfy live qualification"
            kind = "simulated"
        paths = tuple(str(path) for path in record.hermetic_suite_paths)
        normalized.append(
            CohortEvidence(
                case_id=case_id,
                evidence_kind=kind,
                status=status,
                reason=reason,
                live_count=live_count,
                live_environment_id=environment.strip(),
                hermetic_suite_paths=paths,
            )
        )
    _require_unique([item.case_id for item in normalized], REQUIRED_COHORT_CASES, "cohort case")
    by_id = {item.case_id: item for item in normalized}
    return tuple(by_id[case_id] for case_id in REQUIRED_COHORT_CASES)


def _normalize_targets(records: Sequence[TargetMeasurement]) -> tuple[TargetMeasurement, ...]:
    normalized: list[TargetMeasurement] = []
    for record in records:
        target_id = _text(record.target_id, "target_id")
        kind = _kind(record.evidence_kind, "evidence_kind")
        if kind in {"estimated", "simulated"} and (
            record.observed_bps is not None
            or record.observed_count is not None
            or record.observed_net_cost_units is not None
        ):
            raise DirectObjectiveEventDrivenQualificationError(
                f"{target_id} uses {kind} numeric values; record unavailable instead of a fake measurement"
            )
        normalized.append(
            TargetMeasurement(
                target_id=target_id,
                evidence_kind=kind,
                observed_bps=_optional_non_negative_int(record.observed_bps, "observed_bps"),
                observed_count=_optional_non_negative_int(
                    record.observed_count, "observed_count"
                ),
                observed_net_cost_units=_optional_int(
                    record.observed_net_cost_units,
                    "observed_net_cost_units",
                    non_negative=False,
                ),
                reason=_text(record.reason, "reason") if record.reason else kind,
            )
        )
    _require_unique(
        [item.target_id for item in normalized], REQUIRED_TARGETS, "target"
    )
    by_id = {item.target_id: item for item in normalized}
    return tuple(by_id[target_id] for target_id in REQUIRED_TARGETS)


def qualify_direct_objective_event_driven(
    cohort: Sequence[CohortEvidence],
    targets: Sequence[TargetMeasurement],
    safety: SafetyVector,
    *,
    live_campaign_identity: str = "",
) -> QualificationVerdict:
    """Evaluate Phase-0 qualification.  Promotion is fail-closed."""

    normalized_cohort = _normalize_cohort(cohort)
    normalized_targets = _normalize_targets(targets)
    safety_kind = _kind(safety.evidence_kind, "safety.evidence_kind")
    if safety_kind == LIVE_SATISFYING_KIND:
        missing_invariants = [name for name in HARD_ZERO_INVARIANTS if name not in safety.counts]
        if missing_invariants:
            raise DirectObjectiveEventDrivenQualificationError(
                f"live safety vector missing {missing_invariants[0]}"
            )
        for name, count in safety.counts.items():
            if name not in HARD_ZERO_INVARIANTS:
                raise DirectObjectiveEventDrivenQualificationError(f"unknown hard-zero invariant: {name}")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise DirectObjectiveEventDrivenQualificationError(
                    f"{name} must be a non-negative integer"
                )

    blockers: list[str] = []
    missed_live: list[str] = []
    for record in normalized_cohort:
        if record.status == "blocked":
            blockers.append(record.case_id)
        if record.status == "failed":
            blockers.append(record.case_id)
        if not _live_cohort_satisfied(record):
            missed_live.append(record.case_id)
        if record.evidence_kind == "simulated" and record.status == "passed":
            blockers.append("simulated_as_live")

    missed_targets: list[str] = []
    for record in normalized_targets:
        if not _target_satisfied(record):
            missed_targets.append(record.target_id)

    hard_safety_failed = False
    if safety_kind == LIVE_SATISFYING_KIND:
        hard_safety_failed = any(int(safety.counts[name]) > 0 for name in HARD_ZERO_INVARIANTS)
        if hard_safety_failed:
            blockers.append("hard_safety_failures")
    else:
        missed_targets.append("hard_safety_failures") if "hard_safety_failures" not in missed_targets else None

    edits = next(
        item for item in normalized_targets if item.target_id == "manual_task_table_edits"
    )
    if edits.evidence_kind == LIVE_SATISFYING_KIND and edits.observed_count not in (None, 0):
        blockers.append("manual_task_table_edits")

    unique_blockers = tuple(dict.fromkeys(blockers))
    unique_missed_live = tuple(dict.fromkeys(missed_live))
    unique_missed_targets = tuple(dict.fromkeys(missed_targets))

    all_live_passed = not unique_missed_live and not unique_missed_targets and not unique_blockers
    if all_live_passed and live_campaign_identity.strip():
        promotion_status = "supervisor_promoted"
        supervisor_disposition = "supervisor_promoted"
    elif unique_blockers and any(
        item.status == "blocked" for item in normalized_cohort
    ):
        promotion_status = "typed_blocked"
        supervisor_disposition = "supervisor_non_promoted"
    elif unique_missed_live or unique_missed_targets:
        # Honest R&D non-promotion: the live campaign was not qualified.
        promotion_status = "rnd_non_promoted"
        supervisor_disposition = "supervisor_non_promoted"
    else:
        promotion_status = "rnd_non_promoted"
        supervisor_disposition = "supervisor_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise DirectObjectiveEventDrivenQualificationError("internal promotion status is not admitted")
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification must not mint a closed release outcome"
        )

    payload = {
        "schema": QUALIFICATION_VERDICT_SCHEMA,
        "interface": DIRECT_OBJECTIVE_EVENT_DRIVEN_QUALIFICATION_INTERFACE,
        "promotion_status": promotion_status,
        "supervisor_disposition": supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "cohort": [item.to_mapping() for item in normalized_cohort],
        "targets": [item.to_mapping() for item in normalized_targets],
        "safety": safety.to_mapping(),
        "missed_live_cohort": list(unique_missed_live),
        "missed_targets": list(unique_missed_targets),
        "blockers": list(unique_blockers),
        "live_campaign_identity": live_campaign_identity.strip(),
    }
    verdict_cid = content_identity(payload)
    return QualificationVerdict(
        schema=QUALIFICATION_VERDICT_SCHEMA,
        interface=DIRECT_OBJECTIVE_EVENT_DRIVEN_QUALIFICATION_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition=supervisor_disposition,
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        cohort=normalized_cohort,
        targets=normalized_targets,
        safety=safety,
        missed_live_cohort=unique_missed_live,
        missed_targets=unique_missed_targets,
        blockers=unique_blockers,
        verdict_cid=verdict_cid,
        live_campaign_identity=live_campaign_identity.strip(),
    )


def qualify_current_head_without_live_campaign() -> QualificationVerdict:
    """Convenience evaluator for the ordinary missing-live-campaign case."""

    cohort, targets, safety = current_head_unavailable_inputs()
    return qualify_direct_objective_event_driven(cohort, targets, safety)


# Pinned identity of the ordinary missing-live-campaign verdict.  Drift here
# means the default unavailable payload changed and the outer receipt must be
# regenerated from this evaluator rather than transcribed.
CURRENT_HEAD_UNAVAILABLE_VERDICT_CID: Final = (
    "baguqeerani7dpoxz3g2hizjuedw5einvr5p6vbyds2nulirtwfl4w44zrzaa"
)

PCPR_PHASE0_TASK_ID: Final = "PCPR-001"
PCPR_PHASE0_GOAL_ID: Final = "PCPR-G120"


def pcpr_phase0_receipt_promotion(verdict: QualificationVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields.  Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification completion is not authoritative"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise DirectObjectiveEventDrivenQualificationError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise DirectObjectiveEventDrivenQualificationError(
            "promotion_status is not an admitted Phase-0 status"
        )
    return {
        "schema": QUALIFICATION_VERDICT_SCHEMA,
        "interface": DIRECT_OBJECTIVE_EVENT_DRIVEN_QUALIFICATION_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "missed_live_cohort_count": len(verdict.missed_live_cohort),
        "missed_target_count": len(verdict.missed_targets),
        "blockers": list(verdict.blockers),
        "duckdb_or_quack_state_written": False,
        "evidence_kind": "measured",
    }


def current_head_pcpr_phase0_receipt_promotion() -> dict[str, Any]:
    """Fail-closed promotion section for the ordinary missing-live case."""

    return pcpr_phase0_receipt_promotion(qualify_current_head_without_live_campaign())


LIVE_COUNT_CASES: Final[frozenset[str]] = frozenset(
    {
        "ten_consecutive_bounded_objectives",
        "twenty_historical_task_replays",
    }
)

TARGET_RECEIPT_SPEC: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "median_end_to_end_input_token_reduction": {
            "kind": "bps_min",
            "required_bps": TARGET_MEDIAN_INPUT_TOKEN_REDUCTION_BPS,
        },
        "frontier_model_call_reduction": {
            "kind": "bps_min",
            "required_bps": TARGET_FRONTIER_MODEL_CALL_REDUCTION_BPS,
        },
        "net_cost_reduction_after_audit": {
            "kind": "net_cost",
            "required": "positive integer cost units",
        },
        "eligible_decisions_without_frontier_model": {
            "kind": "bps_min",
            "required_bps": TARGET_ELIGIBLE_DECISIONS_WITHOUT_FRONTIER_BPS,
        },
        "ordinary_refills_without_llm": {
            "kind": "bps_min",
            "required_bps": TARGET_ORDINARY_REFILLS_WITHOUT_LLM_BPS,
        },
        "contextpack_reuse_on_eligible_tasks": {
            "kind": "bps_min",
            "required_bps": TARGET_CONTEXTPACK_REUSE_BPS,
        },
        "unnecessary_task_churn": {
            "kind": "bps_max",
            "maximum_bps": TARGET_MAX_UNNECESSARY_TASK_CHURN_BPS,
        },
        "manual_recovery": {
            "kind": "bps_max",
            "maximum_bps": TARGET_MAX_MANUAL_RECOVERY_BPS,
        },
        "manual_task_table_edits": {
            "kind": "count_zero",
            "required_count": TARGET_MANUAL_TASK_TABLE_EDITS,
        },
        "hard_safety_failures": {
            "kind": "count_zero",
            "required_count": TARGET_HARD_SAFETY_FAILURES,
        },
    }
)


def _reject_closed_release_value(value: Any, name: str) -> None:
    if value in CLOSED_RELEASE_OUTCOMES:
        raise DirectObjectiveEventDrivenQualificationError(
            f"{name} must not be a closed PCPR release outcome"
        )


def pcpr_phase0_receipt_live_cohort(verdict: QualificationVerdict) -> dict[str, Any]:
    """Outer-receipt live-cohort section.  Hermetic coverage is not live."""

    cases: list[dict[str, Any]] = []
    for record in verdict.cohort:
        evidence_kind = record.evidence_kind
        live_status = record.status
        if evidence_kind != LIVE_SATISFYING_KIND and live_status == "passed":
            live_status = "unavailable"
        live_evidence = (
            LIVE_SATISFYING_KIND if evidence_kind == LIVE_SATISFYING_KIND else "unavailable"
        )
        item: dict[str, Any] = {
            "case_id": record.case_id,
            "live_status": live_status,
            "evidence_kind": live_evidence,
            "reason": record.reason,
        }
        if record.case_id in LIVE_COUNT_CASES:
            item["live_count"] = (
                record.live_count if evidence_kind == LIVE_SATISFYING_KIND else None
            )
        cases.append(item)
    return {
        "minimum_consecutive_bounded_objectives": LIVE_OBJECTIVE_MINIMUM,
        "minimum_historical_task_replays": LIVE_REPLAY_MINIMUM,
        "live_campaign_executed": bool(verdict.live_campaign_identity)
        and not verdict.missed_live_cohort
        and not verdict.missed_targets
        and not verdict.blockers,
        "manual_database_edits_in_this_task": 0,
        "cases": cases,
    }


def pcpr_phase0_receipt_efficiency_targets(
    verdict: QualificationVerdict,
) -> dict[str, Any]:
    """Outer-receipt efficiency-target section.  Missing values stay null."""

    by_id = {item.target_id: item for item in verdict.targets}
    payload: dict[str, Any] = {}
    for target_id in REQUIRED_TARGETS:
        record = by_id[target_id]
        spec = TARGET_RECEIPT_SPEC[target_id]
        entry: dict[str, Any] = {
            "status": "passed" if _target_satisfied(record) else record.evidence_kind,
            "evidence_kind": record.evidence_kind,
            "reason": record.reason,
        }
        kind = spec["kind"]
        if kind == "bps_min":
            entry["required_bps"] = spec["required_bps"]
            entry["observed_bps"] = (
                record.observed_bps if record.evidence_kind == LIVE_SATISFYING_KIND else None
            )
        elif kind == "bps_max":
            entry["maximum_bps"] = spec["maximum_bps"]
            entry["observed_bps"] = (
                record.observed_bps if record.evidence_kind == LIVE_SATISFYING_KIND else None
            )
        elif kind == "net_cost":
            entry["required"] = spec["required"]
            entry["observed_net_cost_units"] = (
                record.observed_net_cost_units
                if record.evidence_kind == LIVE_SATISFYING_KIND
                else None
            )
        elif kind == "count_zero":
            entry["required_count"] = spec["required_count"]
            entry["observed_count"] = (
                record.observed_count if record.evidence_kind == LIVE_SATISFYING_KIND else None
            )
            if target_id == "manual_task_table_edits":
                entry["this_task_direct_writes"] = 0
        payload[target_id] = entry
    return payload


def pcpr_phase0_receipt_negative_results() -> dict[str, Any]:
    """Fixed negative results for Phase-0 R&D qualification."""

    return {
        "simulated_success_cannot_promote": True,
        "hermetic_pass_cannot_satisfy_live_objective_minimum": True,
        "estimated_token_reduction_rejected": True,
        "missing_metrics_not_recorded_as_zero": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_phase0_receipt_sections() -> dict[str, Any]:
    """Promotion, cohort, target, and negative sections for the missing-live case."""

    verdict = qualify_current_head_without_live_campaign()
    promotion = pcpr_phase0_receipt_promotion(verdict)
    return {
        "qualification_verdict": promotion,
        "required_live_cohort": pcpr_phase0_receipt_live_cohort(verdict),
        "efficiency_targets": pcpr_phase0_receipt_efficiency_targets(verdict),
        "negative_results": pcpr_phase0_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
    }


def validate_pcpr_phase0_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-001 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise DirectObjectiveEventDrivenQualificationError("outer receipt must be a mapping")
    task_id = payload.get("task_id")
    if task_id != PCPR_PHASE0_TASK_ID:
        raise DirectObjectiveEventDrivenQualificationError("outer receipt task_id must be PCPR-001")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise DirectObjectiveEventDrivenQualificationError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise DirectObjectiveEventDrivenQualificationError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification_verdict must be a mapping"
        )
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification_verdict.closed_release_outcome must be null"
        )
    if verdict_section.get("release_claim") is True:
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification must not write DuckDB or Quack state"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise DirectObjectiveEventDrivenQualificationError(
            "qualification_verdict.promotion_status is not an admitted Phase-0 status"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"),
            "acceptance.promotion_status",
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise DirectObjectiveEventDrivenQualificationError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise DirectObjectiveEventDrivenQualificationError(
                "acceptance must not claim a release"
            )
        if acceptance.get("promotion_status") not in {None, promotion_status}:
            raise DirectObjectiveEventDrivenQualificationError(
                "acceptance.promotion_status must match qualification_verdict"
            )

    expected = current_head_pcpr_phase0_receipt_promotion()
    live_campaign = False
    cohort = payload.get("required_live_cohort")
    if isinstance(cohort, Mapping):
        live_campaign = bool(cohort.get("live_campaign_executed"))
    if not live_campaign:
        if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
            raise DirectObjectiveEventDrivenQualificationError(
                "missing-live qualification_verdict.verdict_cid must match the evaluator"
            )
        if promotion_status != expected["promotion_status"]:
            raise DirectObjectiveEventDrivenQualificationError(
                "missing-live promotion_status must match the evaluator"
            )
        if expected["promotion_status"] == "rnd_non_promoted" and promotion_status not in {
            "rnd_non_promoted",
            "typed_unavailable",
            "typed_blocked",
            "supervisor_non_promoted",
        }:
            raise DirectObjectiveEventDrivenQualificationError(
                "missing-live promotion_status must be an honest non-promotion"
            )
    return {
        "valid": True,
        "task_id": PCPR_PHASE0_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": verdict_section.get("verdict_cid"),
        "evidence_kind": "measured",
    }
