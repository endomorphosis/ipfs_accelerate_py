"""EAAEF-142: client disconnect/reconnect against the in-process handoff API.

The run continues in the in-process registry after the client detaches.  Live
sockets and host supervisors are not used.  Reattach requires the exact
``run_id`` and ``authority_id`` issued at admission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import (
    ExternalHandoffAPI,
    ExternalHandoffAPIError,
    ExternalHandoffAuthorityError,
)
from ipfs_accelerate_py.agent_supervisor.api.external_run_handle import ExternalRunHandle
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


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
ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-qualification-artifact@1"
)
PRODUCER_ARGV = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/integration/test_external_client_disconnect.py",
)
RECEIPT_FIELDS = {
    "artifact_cid",
    "authority_id",
    "client_socket",
    "evidence_mode",
    "in_process_run_status",
    "live_eight_container_qualification",
    "live_runtime_invoked",
    "producer_argv",
    "producer_source_cid",
    "production_qualification_claimed",
    "qualification_scope",
    "qualification_status",
    "reattach_requires",
    "registry",
    "run_continued_while_detached",
    "run_id",
    "schema",
    "task_completion_claimed",
    "task_id",
}


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


def _producer_source_cid() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_receipt(payload: dict[str, object]) -> None:
    assert set(payload) == RECEIPT_FIELDS
    assert payload["schema"] == ARTIFACT_SCHEMA
    assert payload["task_id"] == "EAAEF-142"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "in_process_disconnect_contract_only"
    assert payload["qualification_status"] == "not_live_qualified"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_runtime_invoked"] is False
    assert payload["live_eight_container_qualification"] is False
    assert payload["producer_argv"] == list(PRODUCER_ARGV)
    assert payload["producer_source_cid"] == _producer_source_cid()
    unsealed = dict(payload)
    artifact_cid = unsealed.pop("artifact_cid")
    assert artifact_cid == content_identity(unsealed)


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    payload = {
        **payload,
        "producer_argv": list(PRODUCER_ARGV),
        "producer_source_cid": _producer_source_cid(),
    }
    payload["artifact_cid"] = content_identity(payload)
    _validate_receipt(payload)
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
            "schema": ARTIFACT_SCHEMA,
            "task_id": "EAAEF-142",
            "evidence_mode": "contract_fail_closed",
            "qualification_scope": "in_process_disconnect_contract_only",
            "qualification_status": "not_live_qualified",
            "task_completion_claimed": False,
            "production_qualification_claimed": False,
            "live_runtime_invoked": False,
            "live_eight_container_qualification": False,
            "client_socket": False,
            "registry": "in_process",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
            "reattach_requires": ["run_id", "authority_id"],
            "run_continued_while_detached": True,
            "in_process_run_status": terminal.run_status,
        }
    )
    _validate_receipt(payload)
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["live_runtime_invoked"] is False
    assert saved["live_eight_container_qualification"] is False
    assert payload["run_continued_while_detached"] is True
    assert saved["reattach_requires"] == ["run_id", "authority_id"]
