"""Fail-closed regressions for a provider route that launched no provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import provider_failure_policy
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
)


_GROK_SPENDING_LIMIT_MESSAGE = (
    "API error (status 403 Forbidden): personal-team-blocked:spending-limit: "
    "You have run out of credits or need a Grok subscription. Add credits at "
    "https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok."
)


def _legacy_quota_medium_route() -> llm_router.AgentImplementationRoutePlan:
    return llm_router.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.6",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_exhausted",
        fallback_reasoning_effort="medium",
    )


def _capacity_daemon(tmp_path: Path) -> PortalImplementationDaemon:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.state_path = tmp_path / "missing-portal-state.json"
    return daemon


def _route_command(
    *,
    route: llm_router.AgentImplementationRoutePlan,
    nonce: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(
            route.as_binding_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]


def _write_route_log(
    path: Path,
    *,
    receipts: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> None:
    records = [
        *(
            provider_failure_policy.render_grok_failure_receipt(receipt)
            for receipt in receipts
        ),
        *(
            provider_failure_policy.render_grok_route_outcome(outcome)
            for outcome in outcomes
        ),
    ]
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _no_effect_evidence() -> tuple[
    llm_router.AgentImplementationRoutePlan,
    str,
    dict[str, Any],
    dict[str, Any],
]:
    route = _legacy_quota_medium_route()
    nonce = "e" * 64
    receipt = provider_failure_policy.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    outcome = provider_failure_policy.build_grok_route_outcome(
        receipt=receipt,
        route_plan=route.as_binding_dict(),
        decision="denied",
        verifier_status="not_run",
        fallback_dispatched=False,
        fallback_returncode=None,
    )
    return route, nonce, receipt, outcome


def test_exact_legacy_quota_medium_no_effect_denial_is_audit_only(
    tmp_path: Path,
) -> None:
    route, nonce, receipt, outcome = _no_effect_evidence()
    assert route.authorization is None
    assert route.invocation_binding is None
    log_path = tmp_path / "exact-no-effect-route.log"
    _write_route_log(log_path, receipts=[receipt], outcomes=[outcome])

    capacity = _capacity_daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=_route_command(route=route, nonce=nonce),
        returncode=41,
    )

    # A local refund would split Portal accounting from the outer DuckDB
    # attempt. Until a versioned bridge receipt exists, retain exact audit
    # evidence while failing closed on capacity/attempt restoration.
    assert capacity["exhausted"] is False
    assert capacity["providers"] == []
    assert capacity["failure_class"] == "hard_quota_exhausted"
    assert capacity.get("attempt_consumed") is not False
    assert capacity.get("provider_dispatched") is not False
    assert capacity["quota_probe_receipt_id"] == receipt["receipt_id"]
    assert capacity["route_outcome_id"] == outcome["outcome_id"]


@pytest.mark.parametrize(
    "near_miss",
    (
        "malformed_outcome",
        "duplicate_receipt",
        "duplicate_outcome",
        "primary_effect",
        "fallback_effect",
    ),
)
def test_legacy_quota_medium_no_effect_near_misses_fail_closed(
    tmp_path: Path,
    near_miss: str,
) -> None:
    route, nonce, receipt, outcome = _no_effect_evidence()
    receipts = [receipt]
    outcomes = [outcome]
    returncode = 41
    if near_miss == "malformed_outcome":
        outcomes = [{**outcome, "unexpected": "authority"}]
    elif near_miss == "duplicate_receipt":
        receipts = [receipt, receipt]
    elif near_miss == "duplicate_outcome":
        outcomes = [outcome, outcome]
    elif near_miss == "primary_effect":
        receipt = provider_failure_policy.build_grok_failure_receipt(
            probe_stderr_text="Grok Build usage balance exhausted",
            nonce=nonce,
            model="grok-4.6",
            probe_returncode=41,
            primary_dispatched=True,
        )
        outcome = provider_failure_policy.build_grok_route_outcome(
            receipt=receipt,
            route_plan=route.as_binding_dict(),
            decision="denied",
            verifier_status="not_run",
            fallback_dispatched=False,
            fallback_returncode=None,
        )
        receipts = [receipt]
        outcomes = [outcome]
    else:
        outcome = provider_failure_policy.build_grok_route_outcome(
            receipt=receipt,
            route_plan=route.as_binding_dict(),
            decision="fallback_failed",
            verifier_status="not_run",
            fallback_dispatched=True,
            fallback_returncode=17,
        )
        outcomes = [outcome]
        returncode = 17
    log_path = tmp_path / f"{near_miss}.log"
    _write_route_log(log_path, receipts=receipts, outcomes=outcomes)

    capacity = _capacity_daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=_route_command(route=route, nonce=nonce),
        returncode=returncode,
    )

    assert capacity["exhausted"] is False
    assert capacity["providers"] == []
    assert capacity.get("attempt_consumed") is not False
    assert capacity.get("provider_dispatched") is not False


def test_preimplementation_event_distinguishes_raw_kernel_from_bootstrap_override(
    tmp_path: Path,
) -> None:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.implement = True
    daemon._canonical_ref = lambda _task: "sha256:" + ("a" * 64)
    daemon._lgswf_writer_path = lambda _task_id: None
    task = PortalTask(
        task_id="PCTDD-TEST",
        title="exercise bootstrap provider override",
        status="todo",
        completion="automatic",
        priority="P1",
        track="test",
    )

    gate = daemon._evaluate_pre_implementation_provider_gate(
        task=task,
        attempt=1,
        worktree_path=tmp_path,
    )

    assert gate["disposition"] == "abstain_review"
    assert gate["skip_provider"] is False
    assert gate["provider_authorized"] is True
    assert gate["reason_code"] == "no_analytical_close_provider_dispatched"
    event = gate["event"]
    assert event["skip_provider"] is True
    assert event["provider_authorized"] is False
    assert event["reason_code"] == "no_analytical_close"
    assert event["effective_skip_provider"] is False
    assert event["effective_provider_authorized"] is True
    assert (
        event["effective_reason_code"]
        == "no_analytical_close_provider_dispatched"
    )
    assert event["receipt_cid"] == gate["receipt_cid"]
    assert event["kernel_receipt"]["reason_code"] == "no_analytical_close"


def _write_quota_session(
    session: Path,
    *,
    home: Path,
    session_id: str,
) -> None:
    session.mkdir(parents=True)

    def update(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "method": "_x.ai/session/update",
            "params": {"sessionId": session_id, "update": value},
        }

    events = (
        update(
            {
                "sessionUpdate": "retry_state",
                "type": "failed",
                "error_type": "api",
                "message": _GROK_SPENDING_LIMIT_MESSAGE,
            }
        ),
        update(
            {
                "sessionUpdate": "turn_completed",
                "stop_reason": "error",
                "agent_result": _GROK_SPENDING_LIMIT_MESSAGE,
            }
        ),
    )
    transcript = session / "updates.jsonl"
    transcript.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    transcript.chmod(0o600)
    summary = session / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "info": {"id": session_id},
                "current_model_id": "grok-4.6",
                "grok_home": str(home),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary.chmod(0o600)


def _quota_receipt() -> dict[str, Any]:
    return llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="a" * 64,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )


def _encoded_session_parent(home: Path, workspace: Path) -> Path:
    encoded_workspace = quote(
        str(workspace.resolve()),
        safe="!'()*-._~",
    )
    return home / "sessions" / encoded_workspace


def test_quota_evidence_accepts_grok_1_0_13_encoded_workspace_session(
    tmp_path: Path,
) -> None:
    session_id = "f159e13e-462f-43bc-9da2-01bd0c1f5761"
    home = tmp_path / "grok-home"
    workspace = tmp_path / "verifier" / "workspace"
    workspace.mkdir(parents=True)
    _write_quota_session(
        _encoded_session_parent(home, workspace) / session_id,
        home=home,
        session_id=session_id,
    )

    evidence = llm_router.validate_agent_implementation_quota_evidence(
        grok_home=home,
        expected_session_id=session_id,
        verifier_returncode=41,
        failure_receipt=_quota_receipt(),
        verifier_workspace=workspace,
    )

    assert evidence is not None
    assert evidence.verifier_session_id == session_id


def test_quota_evidence_rejects_ambiguous_legacy_and_encoded_sessions(
    tmp_path: Path,
) -> None:
    session_id = "f159e13e-462f-43bc-9da2-01bd0c1f5761"
    home = tmp_path / "grok-home"
    workspace = tmp_path / "verifier" / "workspace"
    workspace.mkdir(parents=True)
    _write_quota_session(
        home / "sessions" / session_id,
        home=home,
        session_id=session_id,
    )
    _write_quota_session(
        _encoded_session_parent(home, workspace) / session_id,
        home=home,
        session_id=session_id,
    )

    assert (
        llm_router.validate_agent_implementation_quota_evidence(
            grok_home=home,
            expected_session_id=session_id,
            verifier_returncode=41,
            failure_receipt=_quota_receipt(),
            verifier_workspace=workspace,
        )
        is None
    )


def test_quota_evidence_rejects_session_for_different_encoded_workspace(
    tmp_path: Path,
) -> None:
    session_id = "f159e13e-462f-43bc-9da2-01bd0c1f5761"
    home = tmp_path / "grok-home"
    expected_workspace = tmp_path / "verifier" / "workspace"
    different_workspace = tmp_path / "different" / "workspace"
    expected_workspace.mkdir(parents=True)
    different_workspace.mkdir(parents=True)
    _write_quota_session(
        _encoded_session_parent(home, different_workspace) / session_id,
        home=home,
        session_id=session_id,
    )

    assert (
        llm_router.validate_agent_implementation_quota_evidence(
            grok_home=home,
            expected_session_id=session_id,
            verifier_returncode=41,
            failure_receipt=_quota_receipt(),
            verifier_workspace=expected_workspace,
        )
        is None
    )


@pytest.mark.parametrize(
    "invalid_state",
    (
        "transcript_symlink",
        "namespace_symlink",
        "workspace_parent_traversal",
        "workspace_symlink",
        "wrong_session",
    ),
)
def test_quota_workspace_layout_rejects_path_or_session_ambiguity(
    tmp_path: Path,
    invalid_state: str,
) -> None:
    session_id = "f159e13e-462f-43bc-9da2-01bd0c1f5761"
    home = tmp_path / "grok-home"
    workspace = tmp_path / "verifier !'()* workspace"
    workspace.mkdir()
    session = _encoded_session_parent(home, workspace) / session_id
    _write_quota_session(
        session,
        home=home,
        session_id=session_id,
    )
    validation_workspace = workspace

    if invalid_state == "transcript_symlink":
        transcript = session / "updates.jsonl"
        saved = transcript.with_suffix(".saved")
        transcript.rename(saved)
        transcript.symlink_to(saved.name)
    elif invalid_state == "namespace_symlink":
        namespace = session.parent
        escaped = tmp_path / "escaped-session-namespace"
        namespace.rename(escaped)
        namespace.symlink_to(escaped, target_is_directory=True)
    elif invalid_state == "workspace_parent_traversal":
        validation_workspace = workspace.parent / "unused" / ".." / workspace.name
    elif invalid_state == "workspace_symlink":
        alias = tmp_path / "workspace-alias"
        alias.symlink_to(workspace, target_is_directory=True)
        validation_workspace = alias
    else:
        wrong_session_id = "7afec563-2424-4fd6-a743-650e86700986"
        transcript = session / "updates.jsonl"
        events = [
            json.loads(line)
            for line in transcript.read_text(encoding="utf-8").splitlines()
        ]
        for event in events:
            event["params"]["sessionId"] = wrong_session_id
        transcript.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )

    assert (
        llm_router.validate_agent_implementation_quota_evidence(
            grok_home=home,
            expected_session_id=session_id,
            verifier_returncode=41,
            failure_receipt=_quota_receipt(),
            verifier_workspace=validation_workspace,
        )
        is None
    )
