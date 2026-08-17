"""Campaign CLI handlers for AAE-056 mutate plan/run/target/explain and report.

Handlers are pure adapters over the AAE public API and leaf planning surfaces.
They never open external repositories, never change production policy, and
fail closed on missing run authority, path exposure, cancellation, and
resource overruns.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, TextIO

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.cli import (
    AssuranceCLIAuthorityError,
    AssuranceCLICancelledError,
    AssuranceCLIError,
    AssuranceCLIResourceError,
    AssuranceCLIUsageError,
    CliCancellationToken,
    CliResourceBudget,
    load_json_mapping,
    load_json_value,
    project_result,
    reject_path_exposure,
    repo_relative_path,
)

# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

CAMPAIGN_CLI_INTERFACE: Final[str] = "AssuranceCampaignCLI@1"
CAMPAIGN_CLI_EVIDENCE: Final[str] = "aae/cli-campaign@1"
REPORT_INTERFACE: Final[str] = "AssuranceCampaignCliReport@1"

MAX_NOTES: Final[int] = 1_024
MAX_REPORT_CANDIDATES: Final[int] = 4_096

CampaignHandler = Callable[..., Mapping[str, Any]]


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


def _enforce_cli_budget_on_plan(
    plan: Mapping[str, Any] | Any,
    budget: CliResourceBudget,
) -> None:
    """Fail closed when a plan exceeds the CLI resource envelope."""

    plan_map: Mapping[str, Any]
    if isinstance(plan, Mapping):
        plan_map = plan
    elif hasattr(plan, "to_dict") and callable(plan.to_dict):
        plan_map = plan.to_dict()
    elif hasattr(plan, "plan") and hasattr(getattr(plan, "plan"), "to_dict"):
        plan_map = plan.plan.to_dict()
    else:
        return

    candidate_cids = plan_map.get("candidate_cids") or plan_map.get("candidates") or ()
    if budget.max_candidates is not None and isinstance(candidate_cids, Sequence):
        if len(candidate_cids) > budget.max_candidates:
            raise AssuranceCLIResourceError(
                "plan candidate count exceeds --max-candidates",
                details={
                    "candidate_count": len(candidate_cids),
                    "max_candidates": budget.max_candidates,
                },
            )

    # Nested budget fields when present.
    nested = plan_map.get("budget") or plan_map.get("resource_budget") or {}
    if isinstance(nested, Mapping):
        if budget.max_worktrees is not None:
            plan_worktrees = nested.get("max_worktrees")
            if plan_worktrees is not None and int(plan_worktrees) > budget.max_worktrees:
                raise AssuranceCLIResourceError(
                    "plan max_worktrees exceeds --max-worktrees",
                    details={
                        "plan_max_worktrees": int(plan_worktrees),
                        "max_worktrees": budget.max_worktrees,
                    },
                )
        if budget.timeout_seconds is not None:
            plan_timeout = nested.get("max_execution_seconds")
            if plan_timeout is not None and int(plan_timeout) > budget.timeout_seconds:
                raise AssuranceCLIResourceError(
                    "plan max_execution_seconds exceeds --timeout-seconds",
                    details={
                        "plan_max_execution_seconds": int(plan_timeout),
                        "timeout_seconds": budget.timeout_seconds,
                    },
                )


def _with_timeout(
    fn: Callable[[], Any],
    *,
    budget: CliResourceBudget,
    cancellation_token: CliCancellationToken | None,
) -> Any:
    """Run *fn* with cooperative cancel checks and a soft wall-clock gate.

    The public AAE APIs are synchronous pure/composed callables; the CLI
    enforces a wall-clock bound around the call and rejects if already
    cancelled. Hard preemption of in-process pure composition is not claimed.
    """

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


def _apply_resource_budget_overrides(
    resource_budget: dict[str, Any],
    budget: CliResourceBudget,
) -> dict[str, Any]:
    """Intersect caller JSON budget with CLI flags (fail-closed min)."""

    data = dict(resource_budget)
    if budget.max_candidates is not None:
        existing = data.get("max_total_candidates")
        if existing is None:
            data["max_total_candidates"] = budget.max_candidates
        else:
            data["max_total_candidates"] = min(int(existing), budget.max_candidates)
    if budget.max_worktrees is not None:
        existing = data.get("max_worktrees")
        if existing is None:
            data["max_worktrees"] = budget.max_worktrees
        else:
            data["max_worktrees"] = min(int(existing), budget.max_worktrees)
    if budget.timeout_seconds is not None:
        existing = data.get("max_execution_seconds")
        if existing is None:
            data["max_execution_seconds"] = budget.timeout_seconds
        else:
            data["max_execution_seconds"] = min(int(existing), budget.timeout_seconds)
    return data


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_mutate_plan(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
) -> Mapping[str, Any]:
    """``assurance mutate plan`` → ``plan_mutation_campaign``."""

    del stdin  # reserved for future stdin envelope support
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    repository_state = load_json_mapping(
        getattr(args, "repository_state_json", None),
        field="repository_state_json",
        budget=active_budget,
    )
    manifest = load_json_mapping(
        getattr(args, "manifest_json", None),
        field="manifest_json",
        budget=active_budget,
    )
    policy = load_json_mapping(
        getattr(args, "policy_json", None),
        field="policy_json",
        budget=active_budget,
    )
    resource_budget = load_json_mapping(
        getattr(args, "resource_budget_json", None),
        field="resource_budget_json",
        budget=active_budget,
    )
    assert repository_state is not None
    assert manifest is not None
    assert policy is not None
    assert resource_budget is not None
    resource_budget = _apply_resource_budget_overrides(resource_budget, active_budget)

    baseline = load_json_value(
        getattr(args, "baseline_json", None),
        field="baseline_json",
        budget=active_budget,
        required=False,
    )
    targets = load_json_value(
        getattr(args, "targets_json", None),
        field="targets_json",
        budget=active_budget,
        required=False,
    )
    operators = load_json_value(
        getattr(args, "operators_json", None),
        field="operators_json",
        budget=active_budget,
        required=False,
    )
    properties = load_json_value(
        getattr(args, "properties_json", None),
        field="properties_json",
        budget=active_budget,
        required=False,
    )
    generation_manifest = load_json_mapping(
        getattr(args, "generation_manifest_json", None),
        field="generation_manifest_json",
        budget=active_budget,
        required=False,
    )
    seed_config = load_json_mapping(
        getattr(args, "seed_config_json", None),
        field="seed_config_json",
        budget=active_budget,
        required=False,
    )

    kwargs: dict[str, Any] = {
        "partition": not bool(getattr(args, "no_partition", False)),
        "notes": _notes(getattr(args, "notes", None)),
        "return_result": True,
    }
    plan_id = getattr(args, "plan_id", None)
    if plan_id:
        kwargs["plan_id"] = str(plan_id)
    if baseline is not None:
        kwargs["baseline"] = baseline
    if targets is not None:
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise AssuranceCLIUsageError(
                "targets_json must be a JSON array",
                reason_code="invalid_targets",
            )
        kwargs["targets"] = list(targets)
    if operators is not None:
        if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)):
            raise AssuranceCLIUsageError(
                "operators_json must be a JSON array",
                reason_code="invalid_operators",
            )
        kwargs["operators"] = list(operators)
    if properties is not None:
        if not isinstance(properties, Sequence) or isinstance(properties, (str, bytes)):
            raise AssuranceCLIUsageError(
                "properties_json must be a JSON array",
                reason_code="invalid_properties",
            )
        kwargs["properties"] = list(properties)
    if generation_manifest is not None:
        kwargs["generation_manifest"] = generation_manifest
    if seed_config is not None:
        kwargs["seed_config"] = seed_config

    campaign_api = _resolve_api(api)

    def _call() -> Any:
        return campaign_api.plan_mutation_campaign(
            repository_state,
            manifest,
            policy,
            resource_budget,
            **kwargs,
        )

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)
    if not isinstance(projected, Mapping):
        projected = {"value": projected}

    # Surface stable identity fields at the top of the CLI result.
    plan_obj = projected.get("plan") if isinstance(projected.get("plan"), Mapping) else projected
    summary = {
        "status": "planned",
        "interface": CAMPAIGN_CLI_INTERFACE,
        "evidence": CAMPAIGN_CLI_EVIDENCE,
        "api": "plan_mutation_campaign",
        "plan_id": plan_obj.get("plan_id") if isinstance(plan_obj, Mapping) else None,
        "plan_cid": plan_obj.get("plan_cid") if isinstance(plan_obj, Mapping) else None,
        "candidate_count": (
            len(plan_obj.get("candidate_cids") or ())
            if isinstance(plan_obj, Mapping)
            else None
        ),
        "resource_budget": active_budget.to_dict(),
        "production_policy_change": False,
        "result": projected,
    }
    return MappingProxyType(summary)


def handle_mutate_run(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
) -> Mapping[str, Any]:
    """``assurance mutate run`` → ``execute_mutation_campaign``.

    Requires explicit ``--authorize-run`` authority.
    """

    del stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    if not bool(getattr(args, "authorize_run", False)):
        raise AssuranceCLIAuthorityError(
            "mutate run requires explicit --authorize-run authority",
            details={"flag": "--authorize-run"},
        )

    plan = load_json_mapping(
        getattr(args, "plan_json", None),
        field="plan_json",
        budget=active_budget,
    )
    verification_policy = load_json_mapping(
        getattr(args, "verification_policy_json", None),
        field="verification_policy_json",
        budget=active_budget,
    )
    assert plan is not None
    assert verification_policy is not None
    _enforce_cli_budget_on_plan(plan, active_budget)

    precomputed_reports = load_json_value(
        getattr(args, "precomputed_reports_json", None),
        field="precomputed_reports_json",
        budget=active_budget,
        required=False,
    )
    candidates = load_json_value(
        getattr(args, "candidates_json", None),
        field="candidates_json",
        budget=active_budget,
        required=False,
    )
    expected_detections = load_json_value(
        getattr(args, "expected_detections_json", None),
        field="expected_detections_json",
        budget=active_budget,
        required=False,
    )

    if precomputed_reports is None and candidates is None:
        # Hermetic default path requires precomputed reports when no executor
        # is injectable through the CLI. Fail closed rather than spawn work.
        raise AssuranceCLIUsageError(
            "mutate run requires --precomputed-reports-json (hermetic execution surface)",
            reason_code="missing_execution_surface",
        )

    if precomputed_reports is not None:
        if not isinstance(precomputed_reports, Sequence) or isinstance(
            precomputed_reports, (str, bytes)
        ):
            raise AssuranceCLIUsageError(
                "precomputed_reports_json must be a JSON array",
                reason_code="invalid_reports",
            )
        if (
            active_budget.max_candidates is not None
            and len(precomputed_reports) > active_budget.max_candidates
        ):
            raise AssuranceCLIResourceError(
                "precomputed reports exceed --max-candidates",
                details={
                    "report_count": len(precomputed_reports),
                    "max_candidates": active_budget.max_candidates,
                },
            )

    kwargs: dict[str, Any] = {"notes": _notes(getattr(args, "notes", None))}
    if precomputed_reports is not None:
        kwargs["precomputed_reports"] = list(precomputed_reports)
    if candidates is not None:
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise AssuranceCLIUsageError(
                "candidates_json must be a JSON array",
                reason_code="invalid_candidates",
            )
        kwargs["candidates"] = list(candidates)
    if expected_detections is not None:
        if not isinstance(expected_detections, Sequence) or isinstance(
            expected_detections, (str, bytes)
        ):
            raise AssuranceCLIUsageError(
                "expected_detections_json must be a JSON array",
                reason_code="invalid_detections",
            )
        kwargs["expected_detections"] = list(expected_detections)

    # Stamp authority metadata (non-authoritative evidence of the CLI gate).
    kwargs["metadata"] = {
        "cli_authorize_run": True,
        "cli_interface": CAMPAIGN_CLI_INTERFACE,
        "production_policy_change": False,
    }

    campaign_api = _resolve_api(api)

    def _call() -> Any:
        return campaign_api.execute_mutation_campaign(
            plan,
            verification_policy,
            **kwargs,
        )

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)
    if not isinstance(projected, Mapping):
        projected = {"value": projected}

    summary = {
        "status": str(projected.get("terminal_status") or "executed"),
        "interface": CAMPAIGN_CLI_INTERFACE,
        "evidence": CAMPAIGN_CLI_EVIDENCE,
        "api": "execute_mutation_campaign",
        "authorized": True,
        "plan_id": projected.get("plan_id"),
        "plan_cid": projected.get("plan_cid"),
        "result_cid": projected.get("result_cid"),
        "killed_count": projected.get("killed_count"),
        "survivor_count": projected.get("survivor_count"),
        "invalid_count": projected.get("invalid_count"),
        "inconclusive_count": projected.get("inconclusive_count"),
        "terminal_status": projected.get("terminal_status"),
        "reason_codes": projected.get("reason_codes"),
        "resource_budget": active_budget.to_dict(),
        "production_policy_change": bool(
            projected.get("production_policy_changed", False)
        ),
        "result": projected,
    }
    if summary["production_policy_change"]:
        raise AssuranceCLIError(
            "campaign claimed production policy change; CLI rejects overclaim",
            reason_code="production_policy_change_forbidden",
        )
    return MappingProxyType(summary)


def handle_mutate_target(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
) -> Mapping[str, Any]:
    """``assurance mutate target`` → ``select_mutation_targets``."""

    del api, stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    properties = load_json_value(
        getattr(args, "properties_json", None),
        field="properties_json",
        budget=active_budget,
    )
    if not isinstance(properties, Sequence) or isinstance(properties, (str, bytes)):
        raise AssuranceCLIUsageError(
            "properties_json must be a JSON array",
            reason_code="invalid_properties",
        )

    repository_id = str(getattr(args, "repository_id", "") or "").strip()
    repository_state_cid = str(getattr(args, "repository_state_cid", "") or "").strip()
    if not repository_id:
        raise AssuranceCLIUsageError(
            "repository_id is required",
            reason_code="missing_repository_id",
        )
    if looks_like_repo_root_token(repository_id):
        raise AssuranceCLIUsageError(
            "repository_id must be an identity token, not a filesystem path",
            reason_code="repository_path_forbidden",
        )
    if not repository_state_cid:
        raise AssuranceCLIUsageError(
            "repository_state_cid is required",
            reason_code="missing_repository_state_cid",
        )
    reject_path_exposure(
        {
            "repository_id": repository_id,
            "repository_state_cid": repository_state_cid,
        },
        path="target_identity",
    )

    sampling_budget = load_json_mapping(
        getattr(args, "sampling_budget_json", None),
        field="sampling_budget_json",
        budget=active_budget,
        required=False,
    )
    if sampling_budget is None and active_budget.max_candidates is not None:
        sampling_budget = {"max_targets": active_budget.max_candidates}

    return_result = bool(getattr(args, "return_result", False))

    from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.targets import (
        select_mutation_targets,
    )

    def _call() -> Any:
        return select_mutation_targets(
            list(properties),
            repository_id=repository_id,
            repository_state_cid=repository_state_cid,
            budget=sampling_budget,
            return_result=return_result,
        )

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)

    if return_result and isinstance(projected, Mapping):
        targets = projected.get("targets") or projected.get("selected_targets") or []
        target_count = len(targets) if isinstance(targets, Sequence) else None
        body = projected
    elif isinstance(projected, Sequence) and not isinstance(projected, (str, bytes)):
        target_count = len(projected)
        body = {"targets": list(projected)}
    else:
        target_count = None
        body = {"value": projected}

    # Scrub any residual source_path host leakage; re-validate relative paths.
    targets_list = body.get("targets") if isinstance(body, Mapping) else None
    if isinstance(targets_list, list):
        cleaned = []
        for item in targets_list:
            if isinstance(item, Mapping):
                item_dict = dict(item)
                source_path = item_dict.get("source_path")
                if isinstance(source_path, str) and source_path:
                    try:
                        item_dict["source_path"] = repo_relative_path(
                            source_path, "source_path"
                        )
                    except Exception:
                        item_dict["source_path"] = "<redacted-host-path>"
                cleaned.append(item_dict)
            else:
                cleaned.append(item)
        body = dict(body)
        body["targets"] = cleaned
        if (
            active_budget.max_candidates is not None
            and len(cleaned) > active_budget.max_candidates
        ):
            raise AssuranceCLIResourceError(
                "selected targets exceed --max-candidates",
                details={
                    "target_count": len(cleaned),
                    "max_candidates": active_budget.max_candidates,
                },
            )

    summary = {
        "status": "selected",
        "interface": CAMPAIGN_CLI_INTERFACE,
        "evidence": CAMPAIGN_CLI_EVIDENCE,
        "api": "select_mutation_targets",
        "repository_id": repository_id,
        "repository_state_cid": repository_state_cid,
        "target_count": target_count,
        "resource_budget": active_budget.to_dict(),
        "production_policy_change": False,
        "result": body,
    }
    return MappingProxyType(summary)


def looks_like_repo_root_token(value: str) -> bool:
    """True when a repository_id looks like a filesystem root rather than an id."""

    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith("/") or text.startswith("\\"):
        return True
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        return True
    if text in {".", ".."} or text.startswith("./") or text.startswith("../"):
        return True
    return False


def handle_mutate_explain(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
) -> Mapping[str, Any]:
    """``assurance mutate explain`` → ``predict_detection_set``."""

    del stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    candidate = load_json_mapping(
        getattr(args, "candidate_json", None),
        field="candidate_json",
        budget=active_budget,
    )
    manifest = load_json_mapping(
        getattr(args, "manifest_json", None),
        field="manifest_json",
        budget=active_budget,
    )
    assert candidate is not None
    assert manifest is not None

    campaign_api = _resolve_api(api)
    notes = _notes(getattr(args, "notes", None))

    def _call() -> Any:
        # Prefer facade method when present; fall back to leaf import.
        if hasattr(campaign_api, "predict_detection_set"):
            return campaign_api.predict_detection_set(
                candidate, manifest, notes=notes
            )
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.planning import (
            predict_detection_set,
        )

        return predict_detection_set(candidate, manifest, notes=notes)

    raw = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    projected = project_result(raw)
    if not isinstance(projected, Mapping):
        projected = {"value": projected}

    detector_count = None
    predictions = projected.get("predictions") or projected.get("detectors")
    if isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes)):
        detector_count = len(predictions)

    summary = {
        "status": "explained",
        "interface": CAMPAIGN_CLI_INTERFACE,
        "evidence": CAMPAIGN_CLI_EVIDENCE,
        "api": "predict_detection_set",
        "candidate_id": projected.get("candidate_id") or candidate.get("candidate_id"),
        "candidate_cid": projected.get("candidate_cid") or candidate.get("candidate_cid"),
        "detector_count": detector_count,
        "resource_budget": active_budget.to_dict(),
        "production_policy_change": False,
        "result": projected,
    }
    return MappingProxyType(summary)


def handle_report(
    args: argparse.Namespace,
    *,
    api: Any | None = None,
    budget: CliResourceBudget | None = None,
    cancellation_token: CliCancellationToken | None = None,
    stdin: TextIO | None = None,
) -> Mapping[str, Any]:
    """``assurance report`` — bounded deterministic campaign report projection.

    Prefer an optional reporting builder when present (AAE-058). Otherwise
    project a closed report from a campaign execution result mapping without
    claiming metrics that are not available.
    """

    del api, stdin
    active_budget = budget or CliResourceBudget()
    _check_cancel(cancellation_token)

    campaign_result = load_json_mapping(
        getattr(args, "campaign_result_json", None),
        field="campaign_result_json",
        budget=active_budget,
    )
    assert campaign_result is not None
    plan = load_json_mapping(
        getattr(args, "plan_json", None),
        field="plan_json",
        budget=active_budget,
        required=False,
    )
    notes = _notes(getattr(args, "notes", None))

    def _call() -> Mapping[str, Any]:
        # Optional AAE-058 builder — fail open to local projection.
        try:
            from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.reporting import (  # type: ignore[attr-defined]
                build_assurance_report,
            )

            built = build_assurance_report(
                campaign_result,
                plan=plan,
                notes=notes,
            )
            return project_result(built)  # type: ignore[return-value]
        except Exception:
            return _project_campaign_report(
                campaign_result,
                plan=plan,
                notes=notes,
                budget=active_budget,
            )

    report = _with_timeout(
        _call, budget=active_budget, cancellation_token=cancellation_token
    )
    if not isinstance(report, Mapping):
        report = {"value": report}

    summary = {
        "status": "reported",
        "interface": REPORT_INTERFACE,
        "evidence": CAMPAIGN_CLI_EVIDENCE,
        "api": "build_assurance_report",
        "plan_id": report.get("plan_id") or campaign_result.get("plan_id"),
        "plan_cid": report.get("plan_cid") or campaign_result.get("plan_cid"),
        "result_cid": report.get("result_cid") or campaign_result.get("result_cid"),
        "report_cid": report.get("report_cid"),
        "terminal_status": report.get("terminal_status")
        or campaign_result.get("terminal_status"),
        "killed_count": report.get("killed_count", campaign_result.get("killed_count")),
        "survivor_count": report.get(
            "survivor_count", campaign_result.get("survivor_count")
        ),
        "invalid_count": report.get(
            "invalid_count", campaign_result.get("invalid_count")
        ),
        "inconclusive_count": report.get(
            "inconclusive_count", campaign_result.get("inconclusive_count")
        ),
        "reason_codes": report.get("reason_codes")
        or campaign_result.get("reason_codes"),
        "summary": report.get("summary"),
        "resource_budget": active_budget.to_dict(),
        "production_policy_change": False,
        "result": report,
    }
    return MappingProxyType(summary)


def _project_campaign_report(
    campaign_result: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None,
    notes: str | None,
    budget: CliResourceBudget,
) -> dict[str, Any]:
    """Deterministic local report projection (no AAE-058 metrics dependency)."""

    reports = campaign_result.get("candidate_reports") or ()
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        reports = ()
    if len(reports) > MAX_REPORT_CANDIDATES:
        raise AssuranceCLIResourceError(
            "candidate_reports exceed report bound",
            details={
                "report_count": len(reports),
                "max": MAX_REPORT_CANDIDATES,
            },
        )
    if budget.max_candidates is not None and len(reports) > budget.max_candidates:
        raise AssuranceCLIResourceError(
            "candidate_reports exceed --max-candidates",
            details={
                "report_count": len(reports),
                "max_candidates": budget.max_candidates,
            },
        )

    # Bounded per-candidate projection (identity + status only).
    projected_reports: list[dict[str, Any]] = []
    for item in reports:
        if not isinstance(item, Mapping):
            continue
        projected_reports.append(
            {
                "candidate_id": item.get("candidate_id"),
                "candidate_cid": item.get("candidate_cid") or item.get("mutant_identity_cid"),
                "terminal_status": item.get("terminal_status")
                or item.get("outcome_status")
                or item.get("disposition"),
                "outcome_cid": item.get("outcome_cid"),
                "report_cid": item.get("report_cid") or item.get("result_cid"),
            }
        )

    killed = campaign_result.get("killed_count")
    survivor = campaign_result.get("survivor_count")
    invalid = campaign_result.get("invalid_count")
    inconclusive = campaign_result.get("inconclusive_count")
    terminal = campaign_result.get("terminal_status")

    plan_id = campaign_result.get("plan_id")
    plan_cid = campaign_result.get("plan_cid")
    if plan is not None:
        plan_id = plan_id or plan.get("plan_id")
        plan_cid = plan_cid or plan.get("plan_cid")

    summary_parts = [
        f"terminal={terminal}",
        f"killed={killed}",
        f"survivor={survivor}",
        f"invalid={invalid}",
        f"inconclusive={inconclusive}",
    ]
    summary_text = " ".join(str(part) for part in summary_parts if part is not None)

    report: dict[str, Any] = {
        "interface": REPORT_INTERFACE,
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "adversarial-assurance-cli-campaign-report@1"
        ),
        "plan_id": plan_id,
        "plan_cid": plan_cid,
        "result_cid": campaign_result.get("result_cid"),
        "repository_state_cid": campaign_result.get("repository_state_cid"),
        "verification_policy_cid": campaign_result.get("verification_policy_cid"),
        "terminal_status": terminal,
        "killed_count": killed,
        "survivor_count": survivor,
        "invalid_count": invalid,
        "inconclusive_count": inconclusive,
        "reason_codes": list(campaign_result.get("reason_codes") or ()),
        "candidate_report_count": len(projected_reports),
        "candidate_reports": projected_reports,
        "require_sandbox": campaign_result.get("require_sandbox", True),
        "network_disabled": campaign_result.get("network_disabled", True),
        "production_policy_changed": False,
        "summary": summary_text,
        "notes": notes,
        "metrics_available": False,
        "metrics_note": (
            "Disjoint campaign metrics (AAE-058) are not claimed by this "
            "CLI projection; counts above are direct execution bindings."
        ),
    }
    reject_path_exposure(report, path="campaign_report")
    return report


# ---------------------------------------------------------------------------
# Handler table
# ---------------------------------------------------------------------------

CAMPAIGN_HANDLERS: Final[dict[str, CampaignHandler]] = {
    "mutate.plan": handle_mutate_plan,
    "mutate.run": handle_mutate_run,
    "mutate.target": handle_mutate_target,
    "mutate.explain": handle_mutate_explain,
    "report": handle_report,
}


def campaign_cli_descriptor() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "interface": CAMPAIGN_CLI_INTERFACE,
            "evidence": CAMPAIGN_CLI_EVIDENCE,
            "commands": list(CAMPAIGN_HANDLERS),
            "report_interface": REPORT_INTERFACE,
            "production_policy_change": False,
            "explicit_run_authority_required": True,
            "apis": {
                "mutate.plan": "plan_mutation_campaign",
                "mutate.run": "execute_mutation_campaign",
                "mutate.target": "select_mutation_targets",
                "mutate.explain": "predict_detection_set",
                "report": "build_assurance_report",
            },
        }
    )


__all__ = [
    "CAMPAIGN_CLI_EVIDENCE",
    "CAMPAIGN_CLI_INTERFACE",
    "CAMPAIGN_HANDLERS",
    "REPORT_INTERFACE",
    "campaign_cli_descriptor",
    "handle_mutate_explain",
    "handle_mutate_plan",
    "handle_mutate_run",
    "handle_mutate_target",
    "handle_report",
    "looks_like_repo_root_token",
]
