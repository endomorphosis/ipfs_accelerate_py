"""Assurance analysis CLI handlers for AAE-057.

Exposes hermetic CLI surfaces for:

* ``assurance gaps`` → ``diagnose_surviving_mutant``
* ``assurance vacuity`` → ``analyze_vacuity``
* ``assurance remediate`` → ``propose_gap_remediation``
* ``assurance evaluate-remediation`` → ``evaluate_remediation``
* ``assurance promote`` → ``promote_assurance_policy`` (requires
  ``--authorize-promote``)
* ``assurance benchmark`` → economics/benchmark projection with honest
  unavailable/inconclusive results when the sealed benchmark surface is
  absent (AAE-062)

Handlers are pure adapters over the AAE public API. They never open external
repositories, never start network services, never change production policy,
preserve candidate versus authority status, and fail closed on missing
promotion authorization, path exposure, cancellation, and resource overruns.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, TextIO

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.cli import (
    ASSURANCE_GROUP,
    CONSOLE_ENTRY,
    EXIT_ERROR,
    EXIT_SUCCESS,
    EXIT_UNAVAILABLE,
    EXIT_USAGE,
    AssuranceCLIAuthorityError,
    AssuranceCLIError,
    AssuranceCLIResourceError,
    AssuranceCLIUsageError,
    CliCancellationToken,
    CliResourceBudget,
    cancellation_from_args,
    emit,
    envelope,
    load_json_mapping,
    load_json_value,
    project_result,
    reject_path_exposure,
    resource_budget_from_args,
)

# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

ASSURANCE_ANALYSIS_CLI_INTERFACE: Final[str] = "AssuranceAnalysisCLI@1"
ASSURANCE_ANALYSIS_CLI_EVIDENCE: Final[str] = "aae/cli-assurance@1"
BENCHMARK_CLI_INTERFACE: Final[str] = "AssuranceBenchmarkCliResult@1"

MAX_NOTES: Final[int] = 1_024
MAX_BENCHMARK_ROWS: Final[int] = 4_096

# Closed vocabulary of dispatch keys for this CLI slice.
ASSURANCE_ANALYSIS_COMMANDS: Final[tuple[str, ...]] = (
    "gaps",
    "vacuity",
    "remediate",
    "evaluate-remediation",
    "promote",
    "benchmark",
)

# Authority vocabulary preserved on every result surface.
AUTHORITY_CANDIDATE: Final[str] = "candidate"
AUTHORITY_AUTHORITY: Final[str] = "authority"
AUTHORITY_NOT_AUTHORITY: Final[str] = "not_authority"
AUTHORITY_INCONCLUSIVE: Final[str] = "inconclusive"

AnalysisHandler = Callable[..., Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _notes(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) > MAX_NOTES:
        raise AssuranceCLIUsageError(
            f"notes exceed {MAX_NOTES} characters",
            reason_code="notes_too_long",
        )
    return text


def _check_cancel(token: CliCancellationToken | None) -> None:
    if token is not None:
        token.check()


def _resolve_api(api: Any | None) -> Any:
    if api is not None:
        return api
    from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api import (
        create_assurance_campaign_api,
    )

    return create_assurance_campaign_api()


def _with_timeout(
    fn: Callable[[], Any],
    *,
    budget: CliResourceBudget,
    cancellation_token: CliCancellationToken | None,
) -> Any:
    """Run *fn* with cooperative cancel checks and a soft wall-clock gate."""

    _check_cancel(cancellation_token)
    started = time.monotonic()
    result = fn()
    _check_cancel(cancellation_token)
    if budget.timeout_seconds is not None and budget.timeout_seconds >= 0:
        elapsed = time.monotonic() - started
        if elapsed > float(budget.timeout_seconds):
            raise AssuranceCLIResourceError(
                "operation exceeded --timeout-seconds",
                details={
                    "elapsed_seconds": round(elapsed, 6),
                    "timeout_seconds": budget.timeout_seconds,
                },
            )
    return result


def _forbid_production_policy_change(
    projected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    claimed = projected.get("production_policy_changed")
    if claimed is None:
        claimed = projected.get("production_policy_change")
    if claimed:
        raise AssuranceCLIError(
            f"{context} claimed production policy change; CLI rejects overclaim",
            reason_code="production_policy_change_forbidden",
        )


def _authority_status_from_result(
    projected: Mapping[str, Any],
    *,
    default: str = AUTHORITY_CANDIDATE,
) -> str:
    """Preserve candidate versus authority status without upgrading drafts."""

    explicit = projected.get("authority_status")
    if isinstance(explicit, str) and explicit.strip():
        token = explicit.strip().lower()
        if token in {
            AUTHORITY_CANDIDATE,
            AUTHORITY_AUTHORITY,
            AUTHORITY_NOT_AUTHORITY,
            AUTHORITY_INCONCLUSIVE,
        }:
            return token

    # Promotion outcomes only become authority when the head actually mutated.
    status = str(projected.get("status") or "").lower()
    head_mutated = bool(projected.get("head_mutated", False))
    if status == "promoted" and head_mutated:
        return AUTHORITY_AUTHORITY
    if status in {"rejected", "conflict", "unavailable", "corrupt", "unchanged"}:
        return AUTHORITY_NOT_AUTHORITY

    # Remediation / evaluation / diagnosis surfaces are never production authority.
    if projected.get("all_heuristic") is True:
        return AUTHORITY_CANDIDATE
    if projected.get("requires_held_out_evaluation") is True:
        return AUTHORITY_CANDIDATE
    candidate_status = projected.get("candidate_status") or projected.get(
        "draft_status"
    )
    if isinstance(candidate_status, str) and "candidate" in candidate_status.lower():
        return AUTHORITY_CANDIDATE
    if projected.get("qualified") is False:
        return AUTHORITY_CANDIDATE
    if str(projected.get("terminal_status") or "").lower() == "inconclusive":
        return AUTHORITY_INCONCLUSIVE
    return default


def _base_summary(
    *,
    status: str,
    api: str,
    projected: Mapping[str, Any],
    budget: CliResourceBudget,
    authority_status: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": status,
        "interface": ASSURANCE_ANALYSIS_CLI_INTERFACE,
        "evidence": ASSURANCE_ANALYSIS_CLI_EVIDENCE,
        "api": api,
        "authority_status": authority_status,
        "candidate_status_preserved": authority_status
        in {AUTHORITY_CANDIDATE, AUTHORITY_INCONCLUSIVE, AUTHORITY_NOT_AUTHORITY}
        or authority_status == AUTHORITY_AUTHORITY,
        "resource_budget": budget.to_dict(),
        "production_policy_change": False,
        "network_service": False,
        "arbitrary_path_exposure": False,
        "result": projected,
    }
    if extra:
        summary.update(dict(extra))
    return summary


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_gaps(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
    gap_repository: Any | None = None,
) -> Mapping[str, Any]:
    """``assurance gaps`` → ``diagnose_surviving_mutant``.

    Diagnoses a surviving mutant and surfaces the sealed assurance gap without
    elevating any candidate remediation to production authority.
    """

    del stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    mutation = load_json_mapping(
        getattr(args, "mutation_json", None),
        field="mutation_json",
        budget=active_budget,
    )
    outcome = load_json_mapping(
        getattr(args, "outcome_json", None),
        field="outcome_json",
        budget=active_budget,
    )
    repository_state = load_json_value(
        getattr(args, "repository_state_json", None),
        field="repository_state_json",
        budget=active_budget,
    )
    assert mutation is not None
    assert outcome is not None
    assert repository_state is not None

    signals = load_json_mapping(
        getattr(args, "signals_json", None),
        field="signals_json",
        budget=active_budget,
        required=False,
    )
    comparison = load_json_mapping(
        getattr(args, "comparison_json", None),
        field="comparison_json",
        budget=active_budget,
        required=False,
    )
    minimized_evidence = load_json_mapping(
        getattr(args, "minimized_evidence_json", None),
        field="minimized_evidence_json",
        budget=active_budget,
        required=False,
    )
    survivor_report = load_json_mapping(
        getattr(args, "survivor_report_json", None),
        field="survivor_report_json",
        budget=active_budget,
        required=False,
    )
    notes = _notes(getattr(args, "notes", None))

    kwargs: dict[str, Any] = {"notes": notes}
    if signals is not None:
        kwargs["signals"] = signals
    if comparison is not None:
        kwargs["comparison"] = comparison
    if minimized_evidence is not None:
        kwargs["minimized_evidence"] = minimized_evidence
    if survivor_report is not None:
        kwargs["survivor_report"] = survivor_report
    if gap_repository is not None:
        kwargs["gap_repository"] = gap_repository
    if bool(getattr(args, "always_persist_gap", False)):
        kwargs["always_persist_gap"] = True

    campaign_api = _resolve_api(api)

    def _call() -> Any:
        if hasattr(campaign_api, "diagnose_surviving_mutant"):
            return campaign_api.diagnose_surviving_mutant(
                mutation, outcome, repository_state, **kwargs
            )
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.diagnosis import (
            diagnose_surviving_mutant,
        )

        return diagnose_surviving_mutant(
            mutation, outcome, repository_state, **kwargs
        )

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)
    if not isinstance(projected, Mapping):
        projected = {"value": projected}
    _forbid_production_policy_change(projected, context="gaps")

    authority = _authority_status_from_result(
        projected, default=AUTHORITY_NOT_AUTHORITY
    )
    # Diagnosis / gap records are evidence, never production authority.
    if authority == AUTHORITY_AUTHORITY:
        authority = AUTHORITY_NOT_AUTHORITY

    gap = projected.get("assurance_gap")
    gap_cid = projected.get("gap_cid")
    if isinstance(gap, Mapping):
        gap_cid = gap_cid or gap.get("gap_cid")

    summary = _base_summary(
        status=str(projected.get("terminal_status") or "diagnosed"),
        api="diagnose_surviving_mutant",
        projected=projected,
        budget=active_budget,
        authority_status=authority,
        extra={
            "candidate_id": projected.get("candidate_id") or mutation.get("candidate_id"),
            "candidate_cid": projected.get("candidate_cid")
            or mutation.get("candidate_cid"),
            "gap_cid": gap_cid,
            "risk_class": projected.get("risk_class"),
            "high_risk": projected.get("high_risk"),
            "requires_human_review": projected.get("requires_human_review"),
            "minimization_failed": projected.get("minimization_failed"),
            "reason_codes": projected.get("reason_codes"),
        },
    )
    return MappingProxyType(summary)


def handle_vacuity(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
) -> Mapping[str, Any]:
    """``assurance vacuity`` → ``analyze_vacuity``.

    Vacuity findings remain analysis evidence and never production authority.
    Residual proof obligations are projected honestly when present.
    """

    del stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    manifest = load_json_mapping(
        getattr(args, "manifest_json", None),
        field="manifest_json",
        budget=active_budget,
    )
    repository_state = load_json_mapping(
        getattr(args, "repository_state_json", None),
        field="repository_state_json",
        budget=active_budget,
    )
    assert manifest is not None
    assert repository_state is not None

    formal_subject = load_json_mapping(
        getattr(args, "formal_subject_json", None),
        field="formal_subject_json",
        budget=active_budget,
        required=False,
    )
    policy_subject = load_json_mapping(
        getattr(args, "policy_subject_json", None),
        field="policy_subject_json",
        budget=active_budget,
        required=False,
    )
    test_subject = load_json_mapping(
        getattr(args, "test_subject_json", None),
        field="test_subject_json",
        budget=active_budget,
        required=False,
    )
    zk_subject = load_json_mapping(
        getattr(args, "zk_receipt_subject_json", None),
        field="zk_receipt_subject_json",
        budget=active_budget,
        required=False,
    )
    subjects = load_json_value(
        getattr(args, "subjects_json", None),
        field="subjects_json",
        budget=active_budget,
        required=False,
    )
    header = load_json_mapping(
        getattr(args, "header_json", None),
        field="header_json",
        budget=active_budget,
        required=False,
    )
    notes = _notes(getattr(args, "notes", None))

    if subjects is not None:
        if not isinstance(subjects, Sequence) or isinstance(subjects, (str, bytes)):
            raise AssuranceCLIUsageError(
                "subjects_json must be a JSON array",
                reason_code="invalid_subjects",
            )
        subjects = list(subjects)

    if not any(
        (formal_subject, policy_subject, test_subject, zk_subject, subjects)
    ):
        raise AssuranceCLIUsageError(
            "vacuity requires at least one subject "
            "(--formal-subject-json, --policy-subject-json, "
            "--test-subject-json, --zk-receipt-subject-json, or --subjects-json)",
            reason_code="missing_vacuity_subjects",
        )

    kwargs: dict[str, Any] = {"notes": notes}
    if formal_subject is not None:
        kwargs["formal_subject"] = formal_subject
    if policy_subject is not None:
        kwargs["policy_subject"] = policy_subject
    if test_subject is not None:
        kwargs["test_subject"] = test_subject
    if zk_subject is not None:
        kwargs["zk_receipt_subject"] = zk_subject
    if subjects is not None:
        kwargs["subjects"] = subjects
    if header is not None:
        kwargs["header"] = header

    campaign_api = _resolve_api(api)

    def _call() -> Any:
        if hasattr(campaign_api, "analyze_vacuity"):
            return campaign_api.analyze_vacuity(manifest, repository_state, **kwargs)
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api import (
            analyze_vacuity,
        )

        return analyze_vacuity(manifest, repository_state, **kwargs)

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)
    if not isinstance(projected, Mapping):
        projected = {"value": projected}
    _forbid_production_policy_change(projected, context="vacuity")

    terminal = str(projected.get("terminal_status") or "analyzed")
    authority = _authority_status_from_result(
        projected, default=AUTHORITY_NOT_AUTHORITY
    )
    if authority == AUTHORITY_AUTHORITY:
        authority = AUTHORITY_NOT_AUTHORITY
    if terminal.lower() == "inconclusive":
        authority = AUTHORITY_INCONCLUSIVE

    findings = projected.get("findings") or ()
    finding_count = (
        len(findings)
        if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes))
        else None
    )

    summary = _base_summary(
        status=terminal,
        api="analyze_vacuity",
        projected=projected,
        budget=active_budget,
        authority_status=authority,
        extra={
            "repository_state_cid": projected.get("repository_state_cid")
            or repository_state.get("repository_state_cid"),
            "assurance_manifest_cid": projected.get("assurance_manifest_cid"),
            "families_analyzed": projected.get("families_analyzed"),
            "finding_count": finding_count,
            "finding_cids": projected.get("finding_cids"),
            "residual_properties": projected.get("residual_properties"),
            "precise_nonclaims": projected.get("precise_nonclaims"),
            "reason_codes": projected.get("reason_codes"),
        },
    )
    return MappingProxyType(summary)


def handle_remediate(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
) -> Mapping[str, Any]:
    """``assurance remediate`` → ``propose_gap_remediation``.

    Proposed remediations remain candidates (heuristic drafts). They never
    self-promote and always require held-out evaluation.
    """

    del stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    surviving_mutant = load_json_mapping(
        getattr(args, "surviving_mutant_json", None),
        field="surviving_mutant_json",
        budget=active_budget,
    )
    assurance_gap = load_json_mapping(
        getattr(args, "assurance_gap_json", None),
        field="assurance_gap_json",
        budget=active_budget,
    )
    assert surviving_mutant is not None
    assert assurance_gap is not None
    notes = _notes(getattr(args, "notes", None))

    campaign_api = _resolve_api(api)

    def _call() -> Any:
        if hasattr(campaign_api, "propose_gap_remediation"):
            return campaign_api.propose_gap_remediation(
                surviving_mutant, assurance_gap, notes=notes
            )
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.remediation import (
            propose_gap_remediation,
        )

        return propose_gap_remediation(
            surviving_mutant, assurance_gap, notes=notes
        )

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)
    if not isinstance(projected, Mapping):
        projected = {"value": projected}
    _forbid_production_policy_change(projected, context="remediate")

    # Force candidate preservation: proposals cannot be authority.
    authority = AUTHORITY_CANDIDATE
    candidate_cids = projected.get("candidate_cids") or ()
    if (
        active_budget.max_candidates is not None
        and isinstance(candidate_cids, Sequence)
        and not isinstance(candidate_cids, (str, bytes))
        and len(candidate_cids) > active_budget.max_candidates
    ):
        raise AssuranceCLIResourceError(
            "proposed candidates exceed --max-candidates",
            details={
                "candidate_count": len(candidate_cids),
                "max_candidates": active_budget.max_candidates,
            },
        )

    summary = _base_summary(
        status="proposed",
        api="propose_gap_remediation",
        projected=projected,
        budget=active_budget,
        authority_status=authority,
        extra={
            "proposal_cid": projected.get("proposal_cid"),
            "plan_cid": projected.get("plan_cid"),
            "gap_cid": projected.get("gap_cid") or assurance_gap.get("gap_cid"),
            "survivor_report_cid": projected.get("survivor_report_cid"),
            "candidate_cids": list(candidate_cids)
            if isinstance(candidate_cids, Sequence)
            and not isinstance(candidate_cids, (str, bytes))
            else [],
            "all_heuristic": projected.get("all_heuristic", True),
            "requires_held_out_evaluation": projected.get(
                "requires_held_out_evaluation", True
            ),
            "candidate_status": "heuristic_candidate",
            "reason_codes": projected.get("reason_codes"),
        },
    )
    return MappingProxyType(summary)


def handle_evaluate_remediation(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
) -> Mapping[str, Any]:
    """``assurance evaluate-remediation`` → ``evaluate_remediation``.

    Qualification remains non-authoritative until a separate authorized
    promotion succeeds. Failures and rejections are reported honestly.
    """

    del stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    remediation = load_json_mapping(
        getattr(args, "remediation_json", None),
        field="remediation_json",
        budget=active_budget,
    )
    held_out_campaign = load_json_mapping(
        getattr(args, "held_out_campaign_json", None),
        field="held_out_campaign_json",
        budget=active_budget,
    )
    assert remediation is not None
    assert held_out_campaign is not None
    notes = _notes(getattr(args, "notes", None))

    max_cost = getattr(args, "max_cost_delta_bp", None)
    report_id = getattr(args, "report_id", None)
    kwargs: dict[str, Any] = {"notes": notes}
    if max_cost is not None:
        kwargs["max_cost_delta_bp"] = int(max_cost)
    if report_id:
        kwargs["report_id"] = str(report_id)
    if bool(getattr(args, "raise_on_hard_reject", False)):
        kwargs["raise_on_hard_reject"] = True

    campaign_api = _resolve_api(api)

    def _call() -> Any:
        if hasattr(campaign_api, "evaluate_remediation"):
            return campaign_api.evaluate_remediation(
                remediation, held_out_campaign, **kwargs
            )
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.remediation import (
            evaluate_remediation,
        )

        return evaluate_remediation(remediation, held_out_campaign, **kwargs)

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)
    if not isinstance(projected, Mapping):
        projected = {"value": projected}
    _forbid_production_policy_change(projected, context="evaluate-remediation")

    qualified = bool(projected.get("qualified", False))
    # Even a held-out pass remains candidate until authorized promotion.
    authority = AUTHORITY_CANDIDATE
    disposition = str(
        projected.get("disposition") or projected.get("verdict") or "evaluated"
    )

    summary = _base_summary(
        status=disposition,
        api="evaluate_remediation",
        projected=projected,
        budget=active_budget,
        authority_status=authority,
        extra={
            "qualified": qualified,
            "verdict": projected.get("verdict"),
            "disposition": projected.get("disposition"),
            "plan_cid": projected.get("plan_cid"),
            "evaluation_report_cid": projected.get("evaluation_report_cid"),
            "qualification_cid": projected.get("qualification_cid"),
            "candidate_cids": projected.get("candidate_cids"),
            "partitions_covered": projected.get("partitions_covered"),
            "missing_partitions": projected.get("missing_partitions"),
            "failed_partitions": projected.get("failed_partitions"),
            "one_mutant_overfit": projected.get("one_mutant_overfit"),
            "mock_bypass": projected.get("mock_bypass"),
            "candidate_status": "heuristic_candidate",
            "reason_codes": projected.get("reason_codes"),
            "rejection_reasons": projected.get("rejection_reasons"),
        },
    )
    return MappingProxyType(summary)


def handle_promote(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
    policy_repository: Any | None = None,
    promotion_repository: Any | None = None,
) -> Mapping[str, Any]:
    """``assurance promote`` → ``promote_assurance_policy``.

    Requires explicit ``--authorize-promote`` authority. Candidates cannot
    self-promote. Without an injected disposable policy repository the API
    returns an honest rejected/unavailable outcome and never mutates a head.
    """

    del stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    if not bool(getattr(args, "authorize_promote", False)):
        raise AssuranceCLIAuthorityError(
            "promote requires explicit --authorize-promote authority",
            reason_code="promote_authority_required",
            details={"flag": "--authorize-promote"},
        )

    remediation = load_json_mapping(
        getattr(args, "remediation_json", None),
        field="remediation_json",
        budget=active_budget,
    )
    evaluation = load_json_mapping(
        getattr(args, "evaluation_json", None),
        field="evaluation_json",
        budget=active_budget,
    )
    campaign_receipt = load_json_mapping(
        getattr(args, "campaign_receipt_json", None),
        field="campaign_receipt_json",
        budget=active_budget,
    )
    promotion_signature = load_json_mapping(
        getattr(args, "promotion_signature_json", None),
        field="promotion_signature_json",
        budget=active_budget,
    )
    assert remediation is not None
    assert evaluation is not None
    assert campaign_receipt is not None
    assert promotion_signature is not None

    authorization = load_json_value(
        getattr(args, "authorization_json", None),
        field="authorization_json",
        budget=active_budget,
        required=False,
    )
    authorization_cid = getattr(args, "authorization_cid", None)
    if authorization is None and authorization_cid:
        authorization = str(authorization_cid).strip()
    if authorization is None or authorization == "":
        raise AssuranceCLIUsageError(
            "promote requires --authorization-json or --authorization-cid",
            reason_code="missing_authorization",
        )
    # Self-promotion check at the CLI gate: auth CID must not equal candidate
    # or evaluation identity when those are present as simple strings.
    auth_token = (
        authorization
        if isinstance(authorization, str)
        else (
            authorization.get("authorization_cid")
            or authorization.get("cid")
            if isinstance(authorization, Mapping)
            else None
        )
    )
    if isinstance(auth_token, str) and auth_token:
        forbidden_ids = {
            remediation.get("candidate_cid"),
            remediation.get("plan_cid"),
            remediation.get("proposal_cid"),
            remediation.get("proposed_policy_cid"),
            evaluation.get("evaluation_report_cid"),
            evaluation.get("report_cid"),
            evaluation.get("qualification_cid"),
        }
        forbidden_ids.discard(None)
        if auth_token in forbidden_ids:
            raise AssuranceCLIAuthorityError(
                "authorization must not equal candidate/evaluation identity "
                "(self-promotion forbidden)",
                reason_code="self_promotion_forbidden",
                details={"authorization_cid": auth_token},
            )

    seal_evidence_cid = str(getattr(args, "seal_evidence_cid", "") or "").strip()
    if not seal_evidence_cid:
        raise AssuranceCLIUsageError(
            "promote requires --seal-evidence-cid",
            reason_code="missing_seal_evidence",
        )
    operation_id = str(getattr(args, "operation_id", "") or "").strip()
    if not operation_id:
        raise AssuranceCLIUsageError(
            "promote requires --operation-id",
            reason_code="missing_operation_id",
        )
    reject_path_exposure(
        {
            "seal_evidence_cid": seal_evidence_cid,
            "operation_id": operation_id,
        },
        path="promote_identity",
    )

    notes = _notes(getattr(args, "notes", None))
    kwargs: dict[str, Any] = {
        "campaign_receipt": campaign_receipt,
        "policy_repository": policy_repository,
        "operation_id": operation_id,
        "promotion_signature": promotion_signature,
        "seal_evidence_cid": seal_evidence_cid,
        "notes": notes,
        "metadata": {
            "cli_authorize_promote": True,
            "cli_interface": ASSURANCE_ANALYSIS_CLI_INTERFACE,
            "production_policy_change": False,
        },
    }
    if promotion_repository is not None:
        kwargs["promotion_repository"] = promotion_repository

    seal_status = getattr(args, "seal_status", None)
    if seal_status:
        kwargs["seal_status"] = str(seal_status)
    workspace = getattr(args, "workspace", None)
    if workspace:
        kwargs["workspace"] = str(workspace)
    for flag, key in (
        ("expected_generation", "expected_generation"),
        ("expected_policy_cid", "expected_policy_cid"),
        ("promoted_policy_cid", "promoted_policy_cid"),
        ("promoted_policy_version", "promoted_policy_version"),
        ("base_policy_cid", "base_policy_cid"),
        ("base_policy_version", "base_policy_version"),
        ("repository_state_cid", "repository_state_cid"),
        ("repository_id", "repository_id"),
    ):
        value = getattr(args, flag, None)
        if value is not None and value != "":
            kwargs[key] = int(value) if key == "expected_generation" else str(value)

    campaign_api = _resolve_api(api)

    def _call() -> Any:
        if hasattr(campaign_api, "promote_assurance_policy"):
            return campaign_api.promote_assurance_policy(
                remediation, evaluation, authorization, **kwargs
            )
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.promotion import (
            promote_assurance_policy,
        )

        return promote_assurance_policy(
            remediation, evaluation, authorization, **kwargs
        )

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)
    if not isinstance(projected, Mapping):
        projected = {"value": projected}
    _forbid_production_policy_change(projected, context="promote")

    status = str(projected.get("status") or "rejected")
    head_mutated = bool(projected.get("head_mutated", False))
    authority = _authority_status_from_result(projected, default=AUTHORITY_NOT_AUTHORITY)
    # Without head mutation, never claim authority.
    if not head_mutated and authority == AUTHORITY_AUTHORITY:
        authority = AUTHORITY_NOT_AUTHORITY

    summary = _base_summary(
        status=status,
        api="promote_assurance_policy",
        projected=projected,
        budget=active_budget,
        authority_status=authority,
        extra={
            "authorized": True,
            "head_mutated": head_mutated,
            "blocking_reasons": projected.get("blocking_reasons"),
            "operation_id": projected.get("operation_id") or operation_id,
            "workspace": projected.get("workspace"),
            "candidate_cid": projected.get("candidate_cid"),
            "evaluation_report_cid": projected.get("evaluation_report_cid"),
            "authorization_cid": projected.get("authorization_cid") or auth_token,
            "promoted_policy_cid": projected.get("promoted_policy_cid"),
            "seal_evidence_cid": projected.get("seal_evidence_cid")
            or seal_evidence_cid,
            "held_out_result": projected.get("held_out_result"),
            "diagnostic": projected.get("diagnostic"),
        },
    )
    return MappingProxyType(summary)


def handle_benchmark(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
    benchmark_runner: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """``assurance benchmark`` — economics/benchmark projection.

    When the sealed AAE-062 benchmark surface is unavailable, returns an
    honest ``unavailable`` / ``inconclusive`` result rather than fabricating
    pass rates or authority. Never starts network services.
    """

    del api, stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    campaign_result = load_json_mapping(
        getattr(args, "campaign_result_json", None),
        field="campaign_result_json",
        budget=active_budget,
        required=False,
    )
    metrics_input = load_json_mapping(
        getattr(args, "metrics_json", None),
        field="metrics_json",
        budget=active_budget,
        required=False,
    )
    notes = _notes(getattr(args, "notes", None))

    def _call() -> Mapping[str, Any]:
        runner = benchmark_runner
        if runner is None:
            # Prefer released AAE-062 benchmark module when present.
            try:
                from benchmarks.agent_supervisor import (  # type: ignore[attr-defined]
                    adversarial_assurance as bench_mod,
                )

                runner = getattr(bench_mod, "run_benchmark", None) or getattr(
                    bench_mod, "benchmark_assurance_campaign", None
                )
            except Exception:
                runner = None
        if runner is None:
            # Optional metrics builder (AAE-058) may still project economics.
            try:
                from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.metrics import (  # type: ignore[attr-defined]
                    build_assurance_metrics,
                )

                runner = build_assurance_metrics
            except Exception:
                runner = None

        if runner is None:
            return {
                "interface": BENCHMARK_CLI_INTERFACE,
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "adversarial-assurance-cli-benchmark@1"
                ),
                "status": "unavailable",
                "available": False,
                "terminal_status": "unavailable",
                "reason_code": "benchmark_unavailable",
                "reason_codes": (
                    "benchmark_unavailable",
                    "no_production_policy_change",
                    "no_arbitrary_path_exposure",
                    "no_network_service",
                ),
                "authority_status": AUTHORITY_NOT_AUTHORITY,
                "metrics_available": False,
                "economics_available": False,
                "fabricated_pass": False,
                "production_policy_changed": False,
                "network_service": False,
                "diagnostic": (
                    "Sealed adversarial-assurance benchmark surface is not "
                    "available in the current tree; result is typed_unavailable "
                    "rather than a fabricated pass."
                ),
                "notes": notes,
                "campaign_result_bound": campaign_result is not None,
                "metrics_input_bound": metrics_input is not None,
            }

        try:
            built = runner(
                campaign_result=campaign_result,
                metrics=metrics_input,
                notes=notes,
            )
        except TypeError:
            # Runner may use positional/simple signatures.
            try:
                built = runner(campaign_result or metrics_input or {})
            except Exception as exc:
                return {
                    "interface": BENCHMARK_CLI_INTERFACE,
                    "status": "inconclusive",
                    "available": False,
                    "terminal_status": "inconclusive",
                    "reason_code": "benchmark_inconclusive",
                    "reason_codes": (
                        "benchmark_inconclusive",
                        "no_production_policy_change",
                    ),
                    "authority_status": AUTHORITY_INCONCLUSIVE,
                    "metrics_available": False,
                    "economics_available": False,
                    "fabricated_pass": False,
                    "production_policy_changed": False,
                    "network_service": False,
                    "diagnostic": f"benchmark runner failed closed: {exc}",
                    "notes": notes,
                }
        except Exception as exc:
            return {
                "interface": BENCHMARK_CLI_INTERFACE,
                "status": "inconclusive",
                "available": False,
                "terminal_status": "inconclusive",
                "reason_code": "benchmark_inconclusive",
                "reason_codes": (
                    "benchmark_inconclusive",
                    "no_production_policy_change",
                ),
                "authority_status": AUTHORITY_INCONCLUSIVE,
                "metrics_available": False,
                "economics_available": False,
                "fabricated_pass": False,
                "production_policy_changed": False,
                "network_service": False,
                "diagnostic": f"benchmark runner failed closed: {exc}",
                "notes": notes,
            }

        projected = project_result(built)
        if not isinstance(projected, Mapping):
            projected = {"value": projected}
        body = dict(projected)
        body.setdefault("interface", BENCHMARK_CLI_INTERFACE)
        body.setdefault("available", True)
        body.setdefault("fabricated_pass", False)
        body.setdefault("production_policy_changed", False)
        body.setdefault("network_service", False)
        body.setdefault("authority_status", AUTHORITY_NOT_AUTHORITY)
        body["notes"] = notes
        # Cap row-like collections.
        for key in ("cases", "rows", "samples", "results"):
            rows = body.get(key)
            if (
                isinstance(rows, Sequence)
                and not isinstance(rows, (str, bytes))
                and len(rows) > MAX_BENCHMARK_ROWS
            ):
                raise AssuranceCLIResourceError(
                    f"benchmark {key} exceed bound",
                    details={"count": len(rows), "max": MAX_BENCHMARK_ROWS},
                )
        if active_budget.max_candidates is not None:
            for key in ("cases", "rows", "samples", "results"):
                rows = body.get(key)
                if (
                    isinstance(rows, Sequence)
                    and not isinstance(rows, (str, bytes))
                    and len(rows) > active_budget.max_candidates
                ):
                    raise AssuranceCLIResourceError(
                        f"benchmark {key} exceed --max-candidates",
                        details={
                            "count": len(rows),
                            "max_candidates": active_budget.max_candidates,
                        },
                    )
        return body

    report = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    if not isinstance(report, Mapping):
        report = {"value": report}
    _forbid_production_policy_change(report, context="benchmark")
    reject_path_exposure(dict(report), path="benchmark_report")

    status = str(report.get("status") or report.get("terminal_status") or "reported")
    available = bool(report.get("available", status not in {"unavailable", "inconclusive"}))
    authority = str(report.get("authority_status") or AUTHORITY_NOT_AUTHORITY)
    if authority == AUTHORITY_AUTHORITY:
        authority = AUTHORITY_NOT_AUTHORITY

    summary = _base_summary(
        status=status,
        api="benchmark_assurance",
        projected=report,
        budget=active_budget,
        authority_status=authority,
        extra={
            "available": available,
            "metrics_available": report.get("metrics_available", available),
            "economics_available": report.get("economics_available", available),
            "fabricated_pass": bool(report.get("fabricated_pass", False)),
            "reason_code": report.get("reason_code"),
            "reason_codes": report.get("reason_codes"),
            "diagnostic": report.get("diagnostic"),
        },
    )
    return MappingProxyType(summary)


# ---------------------------------------------------------------------------
# Handler table / descriptor
# ---------------------------------------------------------------------------

ASSURANCE_HANDLERS: Final[dict[str, AnalysisHandler]] = {
    "gaps": handle_gaps,
    "vacuity": handle_vacuity,
    "remediate": handle_remediate,
    "evaluate-remediation": handle_evaluate_remediation,
    "promote": handle_promote,
    "benchmark": handle_benchmark,
}


def assurance_analysis_cli_descriptor() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "interface": ASSURANCE_ANALYSIS_CLI_INTERFACE,
            "evidence": ASSURANCE_ANALYSIS_CLI_EVIDENCE,
            "commands": list(ASSURANCE_HANDLERS),
            "production_policy_change": False,
            "explicit_promote_authority_required": True,
            "arbitrary_external_repositories": False,
            "network_service": False,
            "preserves_candidate_versus_authority": True,
            "honest_unavailable": True,
            "apis": {
                "gaps": "diagnose_surviving_mutant",
                "vacuity": "analyze_vacuity",
                "remediate": "propose_gap_remediation",
                "evaluate-remediation": "evaluate_remediation",
                "promote": "promote_assurance_policy",
                "benchmark": "benchmark_assurance",
            },
        }
    )


# ---------------------------------------------------------------------------
# Parser registration (parser-only; cold-safe)
# ---------------------------------------------------------------------------


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-json",
        action="store_true",
        default=True,
        help="Emit a deterministic JSON envelope (default).",
    )
    parser.add_argument(
        "--output-human",
        action="store_true",
        help="Emit a bounded human summary instead of JSON.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Hard wall-clock budget for the operation.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Upper bound on candidates admitted for this invocation.",
    )
    parser.add_argument(
        "--max-worktrees",
        type=int,
        default=None,
        help="Upper bound on disposable worktrees for this invocation.",
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Treat the invocation as already cancelled.",
    )
    parser.add_argument(
        "--cancel-file",
        help="If this file exists, cancel cooperatively before dispatch.",
    )


def _add_json_input(
    parser: argparse.ArgumentParser,
    *flags: str,
    dest: str,
    help_text: str,
    required: bool = False,
) -> None:
    parser.add_argument(*flags, dest=dest, required=required, help=help_text)


def register_assurance_analysis_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register gaps/vacuity/remediate/evaluate-remediation/promote/benchmark.

    Intended for composition onto an existing ``assurance`` command group.
    Parser-only and cold-safe: performs no I/O or network activity.
    """

    gaps_p = commands.add_parser(
        "gaps",
        help="Diagnose a surviving mutant and surface assurance gaps.",
    )
    _add_common_flags(gaps_p)
    _add_json_input(
        gaps_p,
        "--mutation-json",
        dest="mutation_json",
        help_text="JSON mutation / DiagnosisMutationBinding mapping.",
    )
    _add_json_input(
        gaps_p,
        "--outcome-json",
        dest="outcome_json",
        help_text="JSON outcome / DiagnosisOutcomeBinding mapping.",
    )
    _add_json_input(
        gaps_p,
        "--repository-state-json",
        dest="repository_state_json",
        help_text="JSON repository_state mapping or CID object.",
    )
    _add_json_input(
        gaps_p,
        "--signals-json",
        dest="signals_json",
        help_text="Optional DiagnosisSignals mapping.",
    )
    _add_json_input(
        gaps_p,
        "--comparison-json",
        dest="comparison_json",
        help_text="Optional DetectionComparisonResult mapping.",
    )
    _add_json_input(
        gaps_p,
        "--minimized-evidence-json",
        dest="minimized_evidence_json",
        help_text="Optional prebuilt MinimizedEvidenceBinding mapping.",
    )
    _add_json_input(
        gaps_p,
        "--survivor-report-json",
        dest="survivor_report_json",
        help_text="Optional prebuilt SurvivingMutantReport mapping.",
    )
    gaps_p.add_argument(
        "--always-persist-gap",
        action="store_true",
        help="Persist a gap even for non-high-risk survivors (when a repository is injected).",
    )
    gaps_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    vacuity_p = commands.add_parser(
        "vacuity",
        help="Analyze formal/policy/test/ZK vacuity with residual proof statements.",
    )
    _add_common_flags(vacuity_p)
    _add_json_input(
        vacuity_p,
        "--manifest-json",
        dest="manifest_json",
        help_text="JSON AssuranceManifest mapping.",
    )
    _add_json_input(
        vacuity_p,
        "--repository-state-json",
        dest="repository_state_json",
        help_text="JSON repository_state mapping.",
    )
    _add_json_input(
        vacuity_p,
        "--formal-subject-json",
        dest="formal_subject_json",
        help_text="Optional formal vacuity subject mapping.",
    )
    _add_json_input(
        vacuity_p,
        "--policy-subject-json",
        dest="policy_subject_json",
        help_text="Optional policy vacuity subject mapping.",
    )
    _add_json_input(
        vacuity_p,
        "--test-subject-json",
        dest="test_subject_json",
        help_text="Optional test vacuity subject mapping.",
    )
    _add_json_input(
        vacuity_p,
        "--zk-receipt-subject-json",
        dest="zk_receipt_subject_json",
        help_text="Optional ZK/receipt vacuity subject mapping.",
    )
    _add_json_input(
        vacuity_p,
        "--subjects-json",
        dest="subjects_json",
        help_text="Optional JSON array of {vacuity_family, subject} mappings.",
    )
    _add_json_input(
        vacuity_p,
        "--header-json",
        dest="header_json",
        help_text="Optional AssuranceArtifactHeader mapping.",
    )
    vacuity_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    remediate_p = commands.add_parser(
        "remediate",
        help="Propose requirement-grounded candidate remediations for one gap.",
    )
    _add_common_flags(remediate_p)
    _add_json_input(
        remediate_p,
        "--surviving-mutant-json",
        dest="surviving_mutant_json",
        help_text="JSON SurvivingMutantReport mapping.",
    )
    _add_json_input(
        remediate_p,
        "--assurance-gap-json",
        dest="assurance_gap_json",
        help_text="JSON AssuranceGap mapping.",
    )
    remediate_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    evaluate_p = commands.add_parser(
        "evaluate-remediation",
        help="Evaluate a remediation against a held-out campaign.",
    )
    _add_common_flags(evaluate_p)
    _add_json_input(
        evaluate_p,
        "--remediation-json",
        dest="remediation_json",
        help_text="JSON remediation plan/proposal/run mapping.",
    )
    _add_json_input(
        evaluate_p,
        "--held-out-campaign-json",
        dest="held_out_campaign_json",
        help_text="JSON HeldOutCampaign mapping with partition results.",
    )
    evaluate_p.add_argument(
        "--max-cost-delta-bp",
        type=int,
        default=None,
        help="Optional max cost delta in basis points.",
    )
    evaluate_p.add_argument(
        "--report-id",
        default=None,
        help="Optional stable evaluation report id.",
    )
    evaluate_p.add_argument(
        "--raise-on-hard-reject",
        action="store_true",
        help="Raise on hard reject reasons instead of sealed reject run.",
    )
    evaluate_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    promote_p = commands.add_parser(
        "promote",
        help="Authorize and CAS-publish an assurance-policy successor.",
    )
    _add_common_flags(promote_p)
    _add_json_input(
        promote_p,
        "--remediation-json",
        dest="remediation_json",
        help_text="JSON canonical remediation / candidate mapping.",
    )
    _add_json_input(
        promote_p,
        "--evaluation-json",
        dest="evaluation_json",
        help_text="JSON held-out evaluation / qualification mapping.",
    )
    _add_json_input(
        promote_p,
        "--authorization-json",
        dest="authorization_json",
        help_text="JSON external authorization binding (or use --authorization-cid).",
    )
    promote_p.add_argument(
        "--authorization-cid",
        default=None,
        help="Authorization CID when not provided via --authorization-json.",
    )
    _add_json_input(
        promote_p,
        "--campaign-receipt-json",
        dest="campaign_receipt_json",
        help_text="JSON AssuranceCampaignReceipt mapping.",
    )
    _add_json_input(
        promote_p,
        "--promotion-signature-json",
        dest="promotion_signature_json",
        help_text="JSON promotion ReceiptSignatureBinding mapping.",
    )
    promote_p.add_argument(
        "--seal-evidence-cid",
        required=True,
        help="Released incremental seal evidence CID.",
    )
    promote_p.add_argument(
        "--operation-id",
        required=True,
        help="Idempotent promotion operation id.",
    )
    promote_p.add_argument(
        "--authorize-promote",
        action="store_true",
        help="Explicit promotion authority. Required (fail closed without it).",
    )
    promote_p.add_argument(
        "--seal-status",
        default=None,
        help="Optional seal availability status (default: released).",
    )
    promote_p.add_argument("--workspace", default=None, help="Policy workspace token.")
    promote_p.add_argument(
        "--expected-generation",
        type=int,
        default=None,
        help="Expected-old policy generation for CAS.",
    )
    promote_p.add_argument(
        "--expected-policy-cid",
        default=None,
        help="Expected-old policy CID for CAS.",
    )
    promote_p.add_argument(
        "--promoted-policy-cid",
        default=None,
        help="Optional explicit promoted policy CID override.",
    )
    promote_p.add_argument(
        "--promoted-policy-version",
        default=None,
        help="Optional promoted policy version token.",
    )
    promote_p.add_argument(
        "--base-policy-cid",
        default=None,
        help="Optional base policy CID override.",
    )
    promote_p.add_argument(
        "--base-policy-version",
        default=None,
        help="Optional base policy version token.",
    )
    promote_p.add_argument(
        "--repository-state-cid",
        default=None,
        help="Optional repository state CID binding.",
    )
    promote_p.add_argument(
        "--repository-id",
        default=None,
        help="Optional repository identity token (not a filesystem path).",
    )
    promote_p.add_argument("--notes", default=None, help="Optional bounded notes.")

    bench_p = commands.add_parser(
        "benchmark",
        help="Project benchmark/economics results with honest unavailable status.",
    )
    _add_common_flags(bench_p)
    _add_json_input(
        bench_p,
        "--campaign-result-json",
        dest="campaign_result_json",
        help_text="Optional campaign execution result mapping.",
    )
    _add_json_input(
        bench_p,
        "--metrics-json",
        dest="metrics_json",
        help_text="Optional precomputed metrics mapping.",
    )
    bench_p.add_argument("--notes", default=None, help="Optional bounded notes.")


def register_assurance_analysis_cli(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register a standalone ``assurance`` group with analysis commands only.

    For product host composition, prefer
    :func:`register_assurance_analysis_commands` on the existing AAE-056 group.
    """

    group = subparsers.add_parser(
        ASSURANCE_GROUP,
        help=(
            "Adversarial assurance analysis CLI "
            "(gaps/vacuity/remediate/evaluate-remediation/promote/benchmark)."
        ),
        description=(
            "Hermetic adversarial-assurance analysis commands. Inputs are typed "
            "JSON identity bindings — never arbitrary external repository roots "
            "or host paths. Promote requires explicit --authorize-promote. "
            "Benchmark returns honest unavailable when sealed economics are "
            "absent. No network service is started."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = group.add_subparsers(
        dest="assurance_command",
        metavar="COMMAND",
        help="Assurance analysis operation.",
    )
    register_assurance_analysis_commands(commands)
    return group


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def resolve_analysis_dispatch_command(args: argparse.Namespace) -> str:
    """Map parsed host args to a closed analysis dispatch key."""

    command = getattr(args, "assurance_command", None)
    if command in ASSURANCE_ANALYSIS_COMMANDS:
        return str(command)
    raise AssuranceCLIUsageError(
        "assurance requires one of: "
        + ", ".join(ASSURANCE_ANALYSIS_COMMANDS),
        reason_code="missing_assurance_command",
    )


def _output_json_mode(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "output_human", False)):
        return False
    return True


def run_assurance_analysis_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    stdin: TextIO = sys.stdin,
    api: Any | None = None,
    cancellation_token: CliCancellationToken | None = None,
    handlers: Mapping[str, Callable[..., Mapping[str, Any]]] | None = None,
    policy_repository: Any | None = None,
    promotion_repository: Any | None = None,
    gap_repository: Any | None = None,
    benchmark_runner: Callable[..., Any] | None = None,
) -> int:
    """Dispatch one analysis command through the assurance handlers / public API."""

    output_json = _output_json_mode(args)
    command = "assurance"
    try:
        command = resolve_analysis_dispatch_command(args)
        budget = resource_budget_from_args(args)
        token = cancellation_from_args(args, token=cancellation_token)
        token.check()

        active_handlers = handlers if handlers is not None else ASSURANCE_HANDLERS
        handler = active_handlers.get(command)
        if handler is None:
            raise AssuranceCLIUsageError(
                f"unknown assurance command: {command}",
                reason_code="unknown_command",
            )

        token.check()
        call_kwargs: dict[str, Any] = {
            "api": api,
            "budget": budget,
            "cancellation_token": token,
            "stdin": stdin,
        }
        if command == "promote":
            call_kwargs["policy_repository"] = policy_repository
            call_kwargs["promotion_repository"] = promotion_repository
        if command == "gaps":
            call_kwargs["gap_repository"] = gap_repository
        if command == "benchmark":
            call_kwargs["benchmark_runner"] = benchmark_runner

        result = handler(args, **call_kwargs)
        if not isinstance(result, Mapping):
            raise AssuranceCLIError(
                "analysis handler returned a non-mapping result",
                reason_code="invalid_handler_result",
            )
        status = str(result.get("status") or "ok")
        # Honest typed unavailable uses exit 3; inconclusive still returns 0
        # with an explicit status so callers can distinguish fabrications.
        if status == "unavailable":
            payload = envelope(
                ok=False,
                command=command,
                status=status,
                result=dict(result),
                exit_code=EXIT_UNAVAILABLE,
                reason_code=str(
                    result.get("reason_code") or "unavailable"
                ),
                error=str(result.get("diagnostic") or "typed_unavailable"),
            )
            emit(payload, output_json=output_json, stream=stdout)
            return EXIT_UNAVAILABLE

        payload = envelope(
            ok=True,
            command=command,
            status=status,
            result=dict(result),
            exit_code=EXIT_SUCCESS,
        )
        emit(payload, output_json=output_json, stream=stdout)
        return EXIT_SUCCESS
    except AssuranceCLIError as exc:
        payload = envelope(
            ok=False,
            command=command,
            status=exc.reason_code,
            error=str(exc),
            reason_code=exc.reason_code,
            exit_code=exc.exit_code,
            details=exc.details,
        )
        emit(payload, output_json=output_json, stream=stdout if output_json else stderr)
        return exc.exit_code
    except Exception as exc:  # map public API errors
        reason = "public_api_error"
        exit_code = EXIT_ERROR
        details: dict[str, Any] = {}
        try:
            from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api import (
                AssuranceApiUnavailableError,
                AssurancePublicApiError,
                PathExposureError,
                UnknownCommandError,
                UnknownFieldError,
            )

            if isinstance(exc, PathExposureError):
                reason = "path_exposure"
                exit_code = EXIT_USAGE
                details = dict(getattr(exc, "details", {}) or {})
            elif isinstance(exc, UnknownCommandError):
                reason = "unknown_command"
                exit_code = EXIT_USAGE
            elif isinstance(exc, UnknownFieldError):
                reason = "unknown_field"
                exit_code = EXIT_USAGE
            elif isinstance(exc, AssuranceApiUnavailableError):
                reason = str(getattr(exc, "reason_code", None) or "unavailable")
                exit_code = EXIT_UNAVAILABLE
                details = {
                    "command": getattr(exc, "command", None),
                    "diagnostic": getattr(exc, "diagnostic", None),
                }
            elif isinstance(exc, AssurancePublicApiError):
                reason = str(getattr(exc, "reason_code", None) or "public_api_error")
                details = dict(getattr(exc, "details", {}) or {})
        except Exception:  # noqa: BLE001
            pass
        if hasattr(exc, "reason_code") and reason == "public_api_error":
            reason = str(getattr(exc, "reason_code") or reason)
        payload = envelope(
            ok=False,
            command=command,
            status=reason,
            error=str(exc)[:1_024],
            reason_code=reason,
            exit_code=exit_code,
            details=details,
        )
        emit(payload, output_json=output_json, stream=stdout if output_json else stderr)
        return exit_code


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    stdin: TextIO = sys.stdin,
    api: Any | None = None,
    cancellation_token: CliCancellationToken | None = None,
    policy_repository: Any | None = None,
    promotion_repository: Any | None = None,
    gap_repository: Any | None = None,
    benchmark_runner: Callable[..., Any] | None = None,
) -> int:
    """Standalone entry for assurance analysis CLI (focused tests)."""

    parser = argparse.ArgumentParser(
        prog=f"{CONSOLE_ENTRY}",
        description="IPFS Accelerate — assurance analysis subset (AAE-057).",
    )
    sub = parser.add_subparsers(dest="command")
    register_assurance_analysis_cli(sub)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else EXIT_USAGE

    if getattr(args, "command", None) != ASSURANCE_GROUP:
        parser.print_help(stderr)
        return EXIT_USAGE
    if not getattr(args, "assurance_command", None):
        stderr.write(
            "usage: ipfs-accelerate assurance "
            "{gaps,vacuity,remediate,evaluate-remediation,promote,benchmark} ...\n"
            "assurance requires an analysis subcommand\n"
        )
        return EXIT_USAGE
    return run_assurance_analysis_cli(
        args,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        api=api,
        cancellation_token=cancellation_token,
        policy_repository=policy_repository,
        promotion_repository=promotion_repository,
        gap_repository=gap_repository,
        benchmark_runner=benchmark_runner,
    )


__all__ = [
    "ASSURANCE_ANALYSIS_CLI_EVIDENCE",
    "ASSURANCE_ANALYSIS_CLI_INTERFACE",
    "ASSURANCE_ANALYSIS_COMMANDS",
    "ASSURANCE_HANDLERS",
    "AUTHORITY_AUTHORITY",
    "AUTHORITY_CANDIDATE",
    "AUTHORITY_INCONCLUSIVE",
    "AUTHORITY_NOT_AUTHORITY",
    "BENCHMARK_CLI_INTERFACE",
    "assurance_analysis_cli_descriptor",
    "handle_benchmark",
    "handle_evaluate_remediation",
    "handle_gaps",
    "handle_promote",
    "handle_remediate",
    "handle_vacuity",
    "main",
    "register_assurance_analysis_cli",
    "register_assurance_analysis_commands",
    "resolve_analysis_dispatch_command",
    "run_assurance_analysis_cli",
]
