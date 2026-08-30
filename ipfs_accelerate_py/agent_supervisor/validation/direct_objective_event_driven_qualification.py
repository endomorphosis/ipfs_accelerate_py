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
    reason: str = (
        "No live PCPR Phase-0 cohort ran in this environment; hermetic suites "
        "remain candidate coverage and are not substituted for live counts or "
        "efficiency targets."
    ),
) -> tuple[tuple[CohortEvidence, ...], tuple[TargetMeasurement, ...], SafetyVector]:
    """Default fail-closed inputs when the live campaign is absent."""

    cohort = tuple(unavailable_cohort_evidence(case_id, reason=reason) for case_id in REQUIRED_COHORT_CASES)
    targets = tuple(unavailable_target_measurement(target_id, reason=reason) for target_id in REQUIRED_TARGETS)
    return cohort, targets, unavailable_safety_vector(reason=reason)


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
    )


def qualify_current_head_without_live_campaign() -> QualificationVerdict:
    """Convenience evaluator for the ordinary missing-live-campaign case."""

    cohort, targets, safety = current_head_unavailable_inputs()
    return qualify_direct_objective_event_driven(cohort, targets, safety)
