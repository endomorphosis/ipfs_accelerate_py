"""EAAEF-142: client disconnect/reconnect against the in-process handoff API.

The run continues in the in-process registry after the client detaches.  Live
sockets and host supervisors are not used.  Reattach requires the exact
``run_id`` and ``authority_id`` issued at admission.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import (
    ExternalHandoffAPI,
    ExternalHandoffAPIError,
    ExternalHandoffAuthorityError,
)
from ipfs_accelerate_py.agent_supervisor.api.external_run_handle import ExternalRunHandle


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "disconnect.json"
)

OPERATOR = "principal:operator"
WORKER = "principal:worker"
REVIEWER = "principal:reviewer"
SESSION = "session:disconnect"
REPO = "repo:disconnect"


def _start_request(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "principal_id": OPERATOR,
        "worker_principal_id": WORKER,
        "session_id": SESSION,
        "repository_id": REPO,
        "objective_id": "objective:disconnect",
        "idempotency_key": "idem:disconnect-1",
    }
    values.update(changes)
    return values


def _bound(started, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "principal_id": OPERATOR,
        "worker_principal_id": WORKER,
        "run_id": started.run_id,
        "authority_id": started.authority_id,
        "session_id": SESSION,
    }
    values.update(changes)
    return values


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_disconnect_pause_run_continues_reattach_requires_run_and_authority() -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(_start_request())
    assert started.verdict == "admitted"
    assert started.run_status == "running"
    handle = ExternalRunHandle.from_receipt(started, api=api)
    steered = handle.steer("continue without the client socket")
    assert steered.run_status == "running"

    paused = api.pause(_bound(started))
    assert paused.run_status == "paused"
    snapshot = handle.detach()
    assert handle.api is None
    assert snapshot["run_id"] == started.run_id
    assert snapshot["authority_id"] == started.authority_id

    # Client socket is gone; the in-process registry still holds the run.
    status_while_detached = api.status(_bound(started))
    assert status_while_detached.run_id == started.run_id
    assert status_while_detached.run_status == "paused"
    assert started.run_id in api._runs

    with pytest.raises(ExternalHandoffAPIError, match="unknown run") as unknown:
        api.attach(_bound(started, run_id=started.authority_id))
    assert unknown.value.reason_code == "unknown_run"
    with pytest.raises(ExternalHandoffAuthorityError) as missing_authority:
        api.attach(_bound(started, authority_id=""))
    assert missing_authority.value.reason_code == "authority_mismatch"
    other = api.handoff(
        _start_request(idempotency_key="idem:other", session_id="session:other")
    )
    with pytest.raises(ExternalHandoffAuthorityError, match="authority id") as mismatch:
        api.attach(_bound(started, authority_id=other.authority_id))
    assert mismatch.value.reason_code == "authority_mismatch"

    attached = api.attach(_bound(started))
    assert attached.reason_code == "attached"
    assert attached.run_id == started.run_id
    assert attached.authority_id == started.authority_id
    resumed = api.resume(_bound(started))
    assert resumed.run_status == "running"

    restored = ExternalRunHandle.deserialize(snapshot, api=api)
    assert restored.run_id == started.run_id
    assert restored.authority_id == started.authority_id
    later = restored.resume_from(started.cursor)
    assert started.cursor not in later.event_ids
    status_after = api.status(_bound(started))
    assert status_after.run_status == "running"

    terminal = api.approve(_bound(started, reviewer_principal_id=REVIEWER))
    assert terminal.run_status == "approved"

    payload = _write_receipt(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-overlay-receipt@1",
            "task_id": "EAAEF-142",
            "evidence_mode": "contract_fail_closed",
            "live_runtime_invoked": False,
            "live_eight_container_qualification": False,
            "client_socket": False,
            "registry": "in_process",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
            "reattach_requires": ["run_id", "authority_id"],
            "run_continued_while_detached": True,
            "terminal_status": terminal.run_status,
        }
    )
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["live_runtime_invoked"] is False
    assert saved["live_eight_container_qualification"] is False
    assert payload["run_continued_while_detached"] is True
    assert saved["reattach_requires"] == ["run_id", "authority_id"]
