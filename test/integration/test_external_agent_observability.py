"""EAAEF-133: observability events bind identities; secrets and unmatched steer fail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import (
    ExternalHandoffAPI,
    ExternalHandoffAuthorityError,
)
from ipfs_accelerate_py.agent_supervisor.observability.external_events import (
    ExternalLifecycleEvent,
    LifecycleEventKind,
    LifecyclePrivacyError,
)
from ipfs_accelerate_py.agent_supervisor.observability.external_metrics import ExternalMetrics


RECEIPT = Path(
    "docs/architecture/external_agent_autonomous_execution_fabric/receipts/observability.json"
)
OPERATOR = "principal:operator"
WORKER = "principal:worker"
ARTIFACT = "sha256:" + ("a" * 64)


def test_events_bind_run_and_reject_secrets() -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(
        {
            "principal_id": OPERATOR,
            "worker_principal_id": WORKER,
            "session_id": "session:obs",
            "repository_id": "repo:obs",
            "objective_id": "objective:obs",
            "idempotency_key": "idem:obs",
        }
    )
    event = ExternalLifecycleEvent(
        kind=LifecycleEventKind.HANDOFF_ACCEPTED,
        sequence=0,
        run_id=started.run_id,
        task_id="task:obs",
        attempt_id="attempt:1",
        fence_token=started.authority_id,
        artifact_cid=ARTIFACT,
        created_at_ms=1_700_000_000_000,
    )
    payload = event.to_dict()
    assert payload["run_id"] == started.run_id
    assert payload["task_id"] == "task:obs"
    assert payload["attempt_id"] == "attempt:1"
    assert payload["fence_token"] == started.authority_id
    assert "api_key" not in payload
    leaked = dict(payload)
    leaked["chain_of_thought"] = "hidden"
    with pytest.raises(LifecyclePrivacyError):
        ExternalLifecycleEvent.from_dict(leaked)


def test_steer_requires_matching_authority() -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(
        {
            "principal_id": OPERATOR,
            "worker_principal_id": WORKER,
            "session_id": "session:obs",
            "repository_id": "repo:obs",
            "objective_id": "objective:obs",
            "idempotency_key": "idem:steer",
        }
    )
    with pytest.raises(ExternalHandoffAuthorityError):
        api.steer(
            {
                "principal_id": OPERATOR,
                "worker_principal_id": WORKER,
                "run_id": started.run_id,
                "authority_id": "authority:forged",
                "instruction": "stop",
            }
        )
    steered = api.steer(
        {
            "principal_id": OPERATOR,
            "worker_principal_id": WORKER,
            "run_id": started.run_id,
            "authority_id": started.authority_id,
            "instruction": "continue",
        }
    )
    assert steered.reason_code == "steered"
    metrics = ExternalMetrics(
        run_id=started.run_id,
        task_id="task:obs",
        observed={"cpu_ms": 3},
        estimated={"remaining_ms": 9},
    )
    assert metrics.as_observation("cpu_ms") == 3
    with pytest.raises(Exception):
        metrics.as_observation("remaining_ms")


def test_observability_receipt() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["task_id"] == "EAAEF-133"
    assert payload["task_alias"] == "EAAEF-133"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "offline_observability_contract_only"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_runtime_invoked"] is False
    assert set(payload["cases"]) == {
        "events_bind_run_task_attempt_fence",
        "secrets_rejected",
        "steer_requires_authority_id",
    }
    assert set(payload["unqualified_requirements"]) == {
        "cursor_continuity_under_restart",
        "pause_resume_cancel_under_restart",
        "bounded_metrics_cardinality",
        "terminal_accounting_under_restart",
    }
